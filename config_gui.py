#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SimpleFEM配置管理器UI
提供图形化界面来配置simple_fem_config.json
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageTk
import numpy as np

class SimpleFEMConfigGUI:
    def __init__(self, root):
        self.root = root

        # 只在非全屏模式下设置窗口大小
        if root.state() != 'zoomed':
            self.root.title("SimpleFEM 配置管理器")
            self.root.geometry("1800x1000")

        # 配置文件路径
        self.config_path = "simple_fem_config.json"
        self.config_data = {}

        # ROI可视化相关
        self.roi1_image = None
        self.roi1_photo = None
        self.current_roi1_path = None

        # Y轴缩放相关
        self.y_zoom_factor = 1.0  # Y轴缩放因子 (1.0 = 原始大小)
        self.y_min_zoom = 0.1    # 最小缩放因子 (10%)
        self.y_max_zoom = 10.0   # 最大缩放因子 (1000%)
        self.y_zoom_step = 0.1   # 每次滚轮缩放步长

        # ROI1图片导航相关
        self.current_image_sequence = []  # 当前序列的图片列表
        self.current_image_index = -1     # 当前图片在序列中的索引

        # 阈值提取测试相关
        self.threshold_lower_var = tk.IntVar(value=140)
        self.threshold_upper_var = tk.IntVar(value=255)
        self.threshold_mask = None
        self.largest_component_mask = None
        self.overlay_enabled = tk.BooleanVar(value=True)
        self.overlay_alpha = tk.DoubleVar(value=0.5)
        self.current_overlay_image = None
        self.current_overlay_photo = None
        self.current_roi3_coords = None
        self.current_mask_for_overlay = None

        # 统计信息显示
        self.total_pixels_var = tk.StringVar(value="0")
        self.largest_component_pixels_var = tk.StringVar(value="0")
        self.component_count_var = tk.StringVar(value="0")

        # Continuous check functionality
        self.continuous_check_enabled = tk.BooleanVar(value=False)

        # 热力图相关状态变量
        self.heat_map = None
        self.heatmap_mode = False
        self.heatmap_alpha_var = tk.DoubleVar(value=0.6)
        self.continuous_heatmap_enabled = tk.BooleanVar(value=False)

        # 创建UI
        self.create_widgets()

        # 加载配置
        self.load_config()

    def create_widgets(self):
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 配置根窗口的行列权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

        row = 0

        # 标题
        title_label = ttk.Label(main_frame, text="SimpleFEM 配置管理器",
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=row, column=0, columnspan=3, pady=(0, 20))
        row += 1

        # 文件操作按钮
        file_frame = ttk.Frame(main_frame)
        file_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Button(file_frame, text="保存配置", command=self.save_config).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(file_frame, text="重新加载", command=self.load_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(file_frame, text="另存为", command=self.save_as).pack(side=tk.LEFT, padx=5)
        row += 1

        # 分隔线
        ttk.Separator(main_frame, orient='horizontal').grid(row=row, column=0, columnspan=3,
                                                            sticky=(tk.W, tk.E), pady=10)
        row += 1

        # 配置选项卡
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        main_frame.rowconfigure(row, weight=1)

        # 基础配置标签页
        self.create_basic_tab()

        # ROI配置标签页
        self.create_roi_tab()

        # 峰值检测标签页
        self.create_peak_detection_tab()

        # ROI3覆盖标签页
        self.create_roi3_override_tab()

        row += 1

        # 底部状态栏
        self.status_var = tk.StringVar(value="就绪")
        status_label = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_label.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))

        # 绑定键盘事件用于ROI1图片导航
        self.root.bind('<Key>', self.on_key_press)
        self.root.bind('<d>', lambda e: self.on_key_press(e))  # 明确绑定D键
        self.root.bind('<a>', lambda e: self.on_key_press(e))  # 明确绑定A键
        self.root.bind('<Left>', lambda e: self.on_key_press(e))  # 明确绑定左箭头
        self.root.bind('<Right>', lambda e: self.on_key_press(e))  # 明确绑定右箭头

        # 确保窗口能接收键盘事件
        print("[DEBUG] 键盘事件已绑定，尝试获取窗口焦点")

        # 延迟设置焦点，确保所有组件都已创建
        self.root.after(100, self.setup_keyboard_focus)

    def setup_keyboard_focus(self):
        """设置键盘焦点"""
        try:
            # 尝试多种方式设置焦点
            self.root.focus_set()
            self.root.grab_set()

            # 尝试将焦点设置到主窗口
            self.root.focus_force()

            print("[DEBUG] 焦点设置完成")

            # 添加焦点丢失检测
            self.root.bind('<FocusIn>', lambda e: print("[DEBUG] 窗口获得焦点"))
            self.root.bind('<FocusOut>', lambda e: print("[DEBUG] 窗口失去焦点"))

        except Exception as e:
            print(f"[ERROR] 焦点设置失败: {e}")

    def create_basic_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="基础配置")

        # 创建滚动框架
        canvas = tk.Canvas(frame)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # 基础配置项
        self.basic_vars = {}

        configs = [
            ("processing_mode", "处理模式", ["screen", "video"]),
            ("data_processing.save_roi1", "保存ROI1图像", "bool"),
            ("data_processing.save_roi2", "保存ROI2图像", "bool"),
            ("data_processing.save_wave", "保存波形图", "bool"),
            ("data_processing.only_delect", "仅保存有波峰的帧", "bool"),
        ]

        row = 0
        for key, label, config_type in configs:
            ttk.Label(scrollable_frame, text=label).grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)

            if config_type == "bool":
                var = tk.BooleanVar()
                ttk.Checkbutton(scrollable_frame, variable=var).grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
            elif key == "processing_mode":
                var = tk.StringVar()
                combo = ttk.Combobox(scrollable_frame, textvariable=var, values=config_type, state="readonly")
                combo.grid(row=row, column=1, sticky=(tk.W, tk.E), padx=5, pady=2)
            else:
                var = tk.StringVar()
                ttk.Entry(scrollable_frame, textvariable=var).grid(row=row, column=1, sticky=(tk.W, tk.E), padx=5, pady=2)

            self.basic_vars[key] = var
            row += 1

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    
    def create_roi_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="ROI配置")

        # 创建三栏布局：左侧-ROI预览，中部-配置，右侧-直方图
        left_frame = ttk.Frame(frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 5), pady=10)

        middle_frame = ttk.Frame(frame)
        middle_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 5), pady=10)

        right_frame = ttk.Frame(frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 10), pady=10)

        self.roi_vars = {}

        # ===== 中部：ROI参数配置 =====

        # 创建中部配置区域
        config_upper_frame = ttk.Frame(middle_frame)
        config_upper_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # ===== 右侧：直方图分析区域 =====
        histogram_frame = ttk.Frame(right_frame)
        histogram_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # ROI1配置
        row = 0
        ttk.Label(config_upper_frame, text="ROI1 大区域配置", font=('Arial', 12, 'bold')).grid(row=row, column=0, columnspan=2, pady=(10, 5))
        row += 1

        roi1_configs = [
            ("roi_capture.default_config.x1", "X1坐标", "int"),
            ("roi_capture.default_config.y1", "Y1坐标", "int"),
            ("roi_capture.default_config.x2", "X2坐标", "int"),
            ("roi_capture.default_config.y2", "Y2坐标", "int"),
        ]

        for key, label, config_type in roi1_configs:
            ttk.Label(config_upper_frame, text=f"  {label}").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
            var = tk.StringVar()
            entry = ttk.Entry(config_upper_frame, textvariable=var, width=15)
            entry.grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
            self.roi_vars[key] = var
            # 绑定值变化事件，实时更新可视化
            var.trace('w', lambda *args: self.update_roi_visualization())
            row += 1

        # ROI2配置
        ttk.Label(config_upper_frame, text="ROI2 小区域配置", font=('Arial', 12, 'bold')).grid(row=row, column=0, columnspan=2, pady=(10, 5))
        row += 1

        roi2_configs = [
            ("roi_capture.roi2_config.extension_params.left", "左边距", "int"),
            ("roi_capture.roi2_config.extension_params.right", "右边距", "int"),
            ("roi_capture.roi2_config.extension_params.top", "上边距", "int"),
            ("roi_capture.roi2_config.extension_params.bottom", "下边距", "int"),
        ]

        for key, label, config_type in roi2_configs:
            ttk.Label(config_upper_frame, text=f"  {label}").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
            var = tk.StringVar()
            entry = ttk.Entry(config_upper_frame, textvariable=var, width=15)
            entry.grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
            self.roi_vars[key] = var
            # 绑定值变化事件，实时更新可视化
            var.trace('w', lambda *args: self.update_roi_visualization())
            row += 1

        # ROI3配置
        ttk.Label(config_upper_frame, text="ROI3 扩展区域配置", font=('Arial', 12, 'bold')).grid(row=row, column=0, columnspan=2, pady=(10, 5))
        row += 1

        roi3_configs = [
            ("roi_capture.roi3_config.extension_params.left", "左边距", "int"),
            ("roi_capture.roi3_config.extension_params.right", "右边距", "int"),
            ("roi_capture.roi3_config.extension_params.top", "上边距", "int"),
            ("roi_capture.roi3_config.extension_params.bottom", "下边距", "int"),
        ]

        for key, label, config_type in roi3_configs:
            ttk.Label(config_upper_frame, text=f"  {label}").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
            var = tk.StringVar()
            entry = ttk.Entry(config_upper_frame, textvariable=var, width=15)
            entry.grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
            self.roi_vars[key] = var
            # 绑定值变化事件，实时更新可视化
            var.trace('w', lambda *args: self.update_roi_visualization())
            row += 1

        # ===== 阈值提取测试区域 =====
        threshold_test_frame = ttk.LabelFrame(config_upper_frame, text="阈值提取测试区域", padding=10)
        threshold_test_frame.grid(row=row, column=0, columnspan=2, sticky=tk.W+tk.E, pady=(15, 5), padx=5)
        row += 1

        # 阈值控制区域
        threshold_control_frame = ttk.Frame(threshold_test_frame)
        threshold_control_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(threshold_control_frame, text="阈值控制:", font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=(0, 10))

        # Lower threshold
        ttk.Label(threshold_control_frame, text="下限:").pack(side=tk.LEFT, padx=(0, 5))
        lower_entry = ttk.Entry(threshold_control_frame, textvariable=self.threshold_lower_var, width=8)
        lower_entry.pack(side=tk.LEFT, padx=(0, 10))

        # Upper threshold
        ttk.Label(threshold_control_frame, text="上限:").pack(side=tk.LEFT, padx=(0, 5))
        upper_entry = ttk.Entry(threshold_control_frame, textvariable=self.threshold_upper_var, width=8)
        upper_entry.pack(side=tk.LEFT, padx=(0, 10))

        # Buttons
        ttk.Button(threshold_control_frame, text="提取", command=self.on_threshold_submit).pack(side=tk.LEFT, padx=2)
        ttk.Button(threshold_control_frame, text="清除", command=self.on_threshold_clear).pack(side=tk.LEFT, padx=2)

        # Continuous check checkbox
        continuous_check_frame = ttk.Frame(threshold_test_frame)
        continuous_check_frame.pack(fill=tk.X, pady=(5, 5))

        self.continuous_check_checkbox = ttk.Checkbutton(
            continuous_check_frame,
            text="连续检查 (加载图片时自动执行提取+最大连通域)",
            variable=self.continuous_check_enabled
        )
        self.continuous_check_checkbox.pack(side=tk.LEFT, padx=(0, 10))

        # 连通域分析区域
        component_frame = ttk.Frame(threshold_test_frame)
        component_frame.pack(fill=tk.X, pady=(5, 5))

        ttk.Label(component_frame, text="连通域分析:", font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(component_frame, text="最大连通域", command=self.on_max_component).pack(side=tk.LEFT, padx=2)
        ttk.Button(component_frame, text="统计", command=self.on_statistics).pack(side=tk.LEFT, padx=2)

        # 统计信息区域
        stats_frame = ttk.Frame(threshold_test_frame)
        stats_frame.pack(fill=tk.X, pady=(5, 5))

        ttk.Label(stats_frame, text="统计信息:", font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(stats_frame, text="总像素:").pack(side=tk.LEFT, padx=(0, 5))
        total_pixels_label = ttk.Label(stats_frame, textvariable=self.total_pixels_var, font=('Courier', 9), relief='sunken', padding=(3, 1))
        total_pixels_label.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(stats_frame, text="最大连通域像素:").pack(side=tk.LEFT, padx=(0, 5))
        largest_pixels_label = ttk.Label(stats_frame, textvariable=self.largest_component_pixels_var, font=('Courier', 9), relief='sunken', padding=(3, 1))
        largest_pixels_label.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(stats_frame, text="连通域数量:").pack(side=tk.LEFT, padx=(0, 5))
        component_count_label = ttk.Label(stats_frame, textvariable=self.component_count_var, font=('Courier', 9), relief='sunken', padding=(3, 1))
        component_count_label.pack(side=tk.LEFT)

        # 叠加控制区域
        overlay_frame = ttk.Frame(threshold_test_frame)
        overlay_frame.pack(fill=tk.X, pady=(5, 0))

        ttk.Label(overlay_frame, text="叠加控制:", font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=(0, 10))

        self.overlay_checkbox = ttk.Checkbutton(overlay_frame, text="显示叠加", variable=self.overlay_enabled,
                                                command=self.on_overlay_toggle)
        self.overlay_checkbox.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(overlay_frame, text="透明度:").pack(side=tk.LEFT, padx=(0, 5))
        self.alpha_scale = ttk.Scale(overlay_frame, from_=0.0, to=1.0, variable=self.overlay_alpha,
                                   orient=tk.HORIZONTAL, length=100, command=self.on_alpha_change)
        self.alpha_scale.pack(side=tk.LEFT, padx=(0, 5))

        self.alpha_value_label = ttk.Label(overlay_frame, text="0.5", font=('Courier', 9))
        self.alpha_value_label.pack(side=tk.LEFT)

        # 热力图控制区域 (在叠加控制区域下方)
        heatmap_frame = ttk.Frame(threshold_test_frame)
        heatmap_frame.pack(fill=tk.X, pady=(5, 0))

        ttk.Label(heatmap_frame, text="热力图控制:", font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=(0, 10))
        self.heatmap_button = ttk.Button(heatmap_frame, text="热力图显示", command=self.on_heatmap_submit)
        self.heatmap_button.pack(side=tk.LEFT, padx=2)
        self.heatmap_clear_button = ttk.Button(heatmap_frame, text="清除热力图", command=self.on_heatmap_clear)
        self.heatmap_clear_button.pack(side=tk.LEFT, padx=2)

        self.continuous_heatmap_checkbox = ttk.Checkbutton(
            heatmap_frame,
            text="连续热力图 (加载图片时自动执行)",
            variable=self.continuous_heatmap_enabled
        )
        self.continuous_heatmap_checkbox.pack(side=tk.LEFT, padx=(10, 5))

        ttk.Label(heatmap_frame, text="热力图透明度:").pack(side=tk.LEFT, padx=(10, 5))
        self.heatmap_alpha_scale = ttk.Scale(heatmap_frame, from_=0.0, to=1.0, variable=self.heatmap_alpha_var,
                                            orient=tk.HORIZONTAL, length=100, command=self.on_heatmap_alpha_change)
        self.heatmap_alpha_scale.pack(side=tk.LEFT, padx=(0, 5))

        self.heatmap_alpha_value_label = ttk.Label(heatmap_frame, text="0.6", font=('Courier', 9))
        self.heatmap_alpha_value_label.pack(side=tk.LEFT)

        # 其他配置
        ttk.Label(config_upper_frame, text="其他配置", font=('Arial', 12, 'bold')).grid(row=row, column=0, columnspan=2, pady=(10, 5))
        row += 1

        other_configs = [
            ("roi_capture.frame_rate", "采集帧率", "int"),
            ("roi3_config.save_roi3", "保存ROI3图像", "bool"),
        ]

        for key, label, config_type in other_configs:
            ttk.Label(config_upper_frame, text=f"  {label}").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)

            if config_type == "bool":
                var = tk.BooleanVar()
                ttk.Checkbutton(config_upper_frame, variable=var).grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
            else:
                var = tk.StringVar()
                ttk.Entry(config_upper_frame, textvariable=var, width=15).grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)

            self.roi_vars[key] = var
            row += 1

        # ===== 直方图分析区域（独立于配置序列）=====

        # 直方图标题
        histogram_title_frame = ttk.Frame(histogram_frame)
        histogram_title_frame.pack(side=tk.TOP, fill=tk.X, pady=(10, 5))

        histogram_title = ttk.Label(histogram_title_frame, text="ROI灰度直方图分析", font=('Arial', 14, 'bold'))
        histogram_title.pack(side=tk.LEFT, padx=10)

        # 直方图画布框架
        curve_frame = ttk.LabelFrame(histogram_frame, text="直方图分析区域", padding=10)
        curve_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(0, 0))

        self.curve_canvas = tk.Canvas(curve_frame, bg='white', width=600, height=900,
                                     highlightthickness=1, highlightbackground='gray')
        self.curve_canvas.pack(fill=tk.BOTH, expand=True)

        # 绑定鼠标滚轮事件用于Y轴缩放
        self.curve_canvas.bind("<MouseWheel>", self.on_curve_canvas_mousewheel)
        self.curve_canvas.bind("<Control-MouseWheel>", self.on_curve_canvas_mousewheel_reset)  # Ctrl+滚轮重置
        self.curve_canvas.bind("<Button-4>", self.on_curve_canvas_mousewheel)  # Linux
        self.curve_canvas.bind("<Button-5>", self.on_curve_canvas_mousewheel)  # Linux
        self.curve_canvas.bind("<Control-Button-4>", self.on_curve_canvas_mousewheel_reset)  # Linux Ctrl+滚轮重置
        self.curve_canvas.bind("<Control-Button-5>", self.on_curve_canvas_mousewheel_reset)  # Linux Ctrl+滚轮重置

        # 为曲线画布添加焦点和键盘事件
        self.curve_canvas.bind('<Button-1>', lambda e: self.curve_canvas.focus_set())  # 点击画布时获取焦点
        self.curve_canvas.bind('<Key>', self.on_key_press)  # 在画布上监听键盘事件
        self.curve_canvas.bind('<d>', lambda e: self.on_key_press(e))
        self.curve_canvas.bind('<a>', lambda e: self.on_key_press(e))

        # ===== 左侧：ROI1预览区域 =====

        # ROI1预览标题和控制
        left_title_frame = ttk.Frame(left_frame)
        left_title_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 5))

        left_title = ttk.Label(left_title_frame, text="ROI1 区域叠加预览", font=('Arial', 14, 'bold'))
        left_title.pack(side=tk.LEFT, padx=10)

        # 图片导入按钮放在标题右侧
        button_frame = ttk.Frame(left_title_frame)
        button_frame.pack(side=tk.RIGHT, padx=10)

        ttk.Button(button_frame, text="导入图片",
                  command=self.import_roi1_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="清除图片",
                  command=self.clear_roi1_image).pack(side=tk.LEFT, padx=5)

        # 当前图片路径显示
        self.image_path_var = tk.StringVar(value="未选择图片")
        path_frame = ttk.Frame(left_frame)
        path_frame.pack(side=tk.TOP, fill=tk.X, pady=(5, 10))

        path_label = ttk.Label(path_frame, text="当前图片:", font=('Arial', 10))
        path_label.pack(side=tk.LEFT, padx=(10, 5))

        path_value = ttk.Label(path_frame, textvariable=self.image_path_var,
                              foreground='gray', font=('Arial', 9))
        path_value.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # ROI1预览画布框架
        roi_frame = ttk.LabelFrame(left_frame, text="ROI1预览区域", padding=10)
        roi_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.roi_canvas = tk.Canvas(roi_frame, bg='white', width=640, height=900,
                                   highlightthickness=1, highlightbackground='gray')
        self.roi_canvas.pack(fill=tk.BOTH, expand=True)

        # 绑定ROI1画布鼠标事件，用于显示灰度值
        self.roi_canvas.bind('<Motion>', self.on_roi_canvas_mouse_motion)
        self.roi_canvas.bind('<Leave>', self.on_roi_canvas_mouse_leave)

        # 在ROI1画布下方添加固定的像素信息显示框
        pixel_info_frame = ttk.Frame(roi_frame)
        pixel_info_frame.pack(fill=tk.X, pady=(5, 0))

        # 像素信息标签
        info_label = ttk.Label(pixel_info_frame, text="鼠标位置信息:", font=('Arial', 10, 'bold'))
        info_label.pack(side=tk.LEFT, padx=(0, 10))

        # 像素信息显示文本框
        self.pixel_info_var = tk.StringVar(value="请导入图片并在ROI1区域移动鼠标")
        self.pixel_info_label = ttk.Label(pixel_info_frame, textvariable=self.pixel_info_var,
                                         font=('Courier', 9), foreground='blue', relief='sunken',
                                         padding=(5, 2))
        self.pixel_info_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # 底部：图例说明 (放在直方图区域下方)
        legend_container = ttk.Frame(histogram_frame)
        legend_container.pack(fill=tk.X, pady=(5, 0))

        # ROI图例
        roi_legend_frame = ttk.LabelFrame(legend_container, text="ROI区域图例", padding=5)
        roi_legend_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        roi_legend_content = ttk.Frame(roi_legend_frame)
        roi_legend_content.pack()

        ttk.Label(roi_legend_content, text="ROI区域:", font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=5)

        # ROI1图例（背景）
        roi1_legend = tk.Canvas(roi_legend_content, width=25, height=15, bg='white')
        roi1_legend.pack(side=tk.LEFT, padx=2)
        roi1_legend.create_rectangle(2, 2, 23, 13, fill='', outline='darkgreen', width=2, dash=(3, 1))
        ttk.Label(roi_legend_content, text="ROI1(背景)", font=('Arial', 9)).pack(side=tk.LEFT, padx=(0, 10))

        # ROI2图例
        roi2_legend = tk.Canvas(roi_legend_content, width=25, height=15, bg='white')
        roi2_legend.pack(side=tk.LEFT, padx=2)
        roi2_legend.create_rectangle(2, 2, 23, 13, fill='', outline='red', width=2)
        ttk.Label(roi_legend_content, text="ROI2", font=('Arial', 9)).pack(side=tk.LEFT, padx=(0, 10))

        # ROI3图例
        roi3_legend = tk.Canvas(roi_legend_content, width=25, height=15, bg='white')
        roi3_legend.pack(side=tk.LEFT, padx=2)
        roi3_legend.create_rectangle(2, 2, 23, 13, fill='', outline='blue', width=2, dash=(4, 2))
        ttk.Label(roi_legend_content, text="ROI3", font=('Arial', 9)).pack(side=tk.LEFT, padx=(0, 10))

        # 交点图例
        intersection_legend = tk.Canvas(roi_legend_content, width=25, height=15, bg='white')
        intersection_legend.pack(side=tk.LEFT, padx=2)
        intersection_legend.create_oval(7, 5, 18, 11, fill='lime', outline='darkgreen')
        ttk.Label(roi_legend_content, text="中心点", font=('Arial', 9)).pack(side=tk.LEFT, padx=(0, 10))

        # 灰度曲线图例
        curve_legend_frame = ttk.LabelFrame(legend_container, text="灰度直方图图例", padding=5)
        curve_legend_frame.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(10, 0))

        curve_legend_content = ttk.Frame(curve_legend_frame)
        curve_legend_content.pack()

        ttk.Label(curve_legend_content, text="直方图:", font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=5)

        # ROI2灰度曲线图例
        roi2_curve_legend = tk.Canvas(curve_legend_content, width=25, height=15, bg='white')
        roi2_curve_legend.pack(side=tk.LEFT, padx=2)
        roi2_curve_legend.create_line(2, 12, 23, 3, fill='red', width=2)
        ttk.Label(curve_legend_content, text="ROI2像素分布", font=('Arial', 9)).pack(side=tk.LEFT, padx=(0, 10))

        # ROI3灰度曲线图例
        roi3_curve_legend = tk.Canvas(curve_legend_content, width=25, height=15, bg='white')
        roi3_curve_legend.pack(side=tk.LEFT, padx=2)
        roi3_curve_legend.create_line(2, 12, 23, 3, fill='blue', width=2)
        ttk.Label(curve_legend_content, text="ROI3像素分布", font=('Arial', 9)).pack(side=tk.LEFT, padx=(0, 10))

        # 移除阈值线图例 - 灰度直方图不需要显示阈值线
        # threshold_legend = tk.Canvas(curve_legend_content, width=25, height=15, bg='white')
        # threshold_legend.pack(side=tk.LEFT, padx=2)
        # threshold_legend.create_line(2, 8, 23, 8, fill='green', width=2, dash=(5, 2))
        # ttk.Label(curve_legend_content, text="检测阈值", font=('Arial', 9)).pack(side=tk.LEFT, padx=(0, 10))

        # 坐标轴说明
        axis_label = ttk.Label(curve_legend_content, text="X:灰度值(0-255) | Y:像素数", font=('Arial', 8), foreground='gray')
        axis_label.pack(side=tk.LEFT, padx=(15, 0))

  
    def create_peak_detection_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="峰值检测")

        self.peak_vars = {}

        configs = [
            ("peak_detection.threshold", "固定阈值", "float"),
            ("peak_detection.adaptive_threshold_enabled", "启用自适应阈值", "bool"),
            ("peak_detection.threshold_over_mean_ratio", "自适应阈值上浮比例", "float"),
            ("peak_detection.difference_threshold", "绿红分类阈值", "float"),
            ("peak_detection.margin_frames", "峰间最小间隔(帧)", "int"),
            ("peak_detection.silence_frames", "干净区间长度(帧)", "int"),
            ("peak_detection.min_region_length", "最小波峰宽度(帧)", "int"),
            ("peak_detection.pre_post_avg_frames", "平均值窗口帧数", "int"),
            ("peak_detection.adaptive_window_seconds", "自适应时间窗口(秒)", "float"),
        ]

        row = 0
        for key, label, config_type in configs:
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky=tk.W, padx=5, pady=3)

            if config_type == "bool":
                var = tk.BooleanVar()
                ttk.Checkbutton(frame, variable=var).grid(row=row, column=1, sticky=tk.W, padx=5, pady=3)
            else:
                var = tk.StringVar()
                ttk.Entry(frame, textvariable=var, width=20).grid(row=row, column=1, sticky=tk.W, padx=5, pady=3)

            self.peak_vars[key] = var
            row += 1

        # 添加说明
        help_text = """说明:
- 固定阈值: 基础的灰度阈值
- 自适应阈值: 阈值=最近N秒平均值*(1+上浮比例)
- 绿红分类阈值: 峰后平均值 - 峰前平均值 > 阈值则为绿色，否则红色
- 峰间最小间隔: 两个波峰间隔小于此值只保留峰值更高的
- 干净区间长度: 波峰前后必须连续低于阈值的帧数"""

        help_label = ttk.Label(frame, text=help_text, justify=tk.LEFT,
                               font=('Arial', 9), foreground='gray')
        help_label.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=5, pady=10)

    def create_roi3_override_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="ROI3覆盖")

        self.roi3_override_vars = {}

        # 主配置
        ttk.Label(frame, text="ROI3覆盖逻辑配置", font=('Arial', 14, 'bold')).grid(row=0, column=0, columnspan=3, pady=(10, 15))

        # 启用开关
        row = 1
        ttk.Label(frame, text="启用ROI3覆盖功能:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=8)
        enabled_var = tk.BooleanVar()
        ttk.Checkbutton(frame, variable=enabled_var,
                       command=self.on_roi3_override_toggle).grid(row=row, column=1, sticky=tk.W, padx=5, pady=8)
        self.roi3_override_vars["peak_detection.roi3_override.enabled"] = enabled_var
        row += 1

        # 分隔线
        ttk.Separator(frame, orient='horizontal').grid(row=row, column=0, columnspan=3,
                                                      sticky=(tk.W, tk.E), pady=10)
        row += 1

        # 阈值设置
        ttk.Label(frame, text="ROI3峰值阈值:", font=('Arial', 11, 'bold')).grid(row=row, column=0, sticky=tk.W, padx=5, pady=5)
        threshold_var = tk.StringVar()
        threshold_entry = ttk.Entry(frame, textvariable=threshold_var, width=15)
        threshold_entry.grid(row=row, column=1, sticky=tk.W, padx=5, pady=5)
        self.roi3_override_vars["peak_detection.roi3_override.threshold"] = threshold_var

        # 添加阈值说明
        ttk.Label(frame, text="(当ROI3峰值大于此值时，红色波峰将被覆盖为绿色)",
                 font=('Arial', 9), foreground='blue').grid(row=row, column=2, sticky=tk.W, padx=5, pady=5)
        row += 1

        # 要求ROI3数据
        ttk.Label(frame, text="要求有效的ROI3数据:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=8)
        require_var = tk.BooleanVar()
        ttk.Checkbutton(frame, variable=require_var).grid(row=row, column=1, sticky=tk.W, padx=5, pady=8)
        self.roi3_override_vars["peak_detection.roi3_override.require_roi3_data"] = require_var
        row += 1

        # 预设值按钮
        ttk.Label(frame, text="快速设置:", font=('Arial', 11, 'bold')).grid(row=row, column=0, sticky=tk.W, padx=5, pady=10)
        row += 1

        preset_frame = ttk.Frame(frame)
        preset_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)

        ttk.Button(preset_frame, text="保守 (100)",
                  command=lambda: self.set_roi3_threshold(100)).pack(side=tk.LEFT, padx=3)
        ttk.Button(preset_frame, text="默认 (115)",
                  command=lambda: self.set_roi3_threshold(115)).pack(side=tk.LEFT, padx=3)
        ttk.Button(preset_frame, text="积极 (130)",
                  command=lambda: self.set_roi3_threshold(130)).pack(side=tk.LEFT, padx=3)
        ttk.Button(preset_frame, text="极高 (150)",
                  command=lambda: self.set_roi3_threshold(150)).pack(side=tk.LEFT, padx=3)
        row += 1

        # 分隔线
        ttk.Separator(frame, orient='horizontal').grid(row=row, column=0, columnspan=3,
                                                      sticky=(tk.W, tk.E), pady=10)
        row += 1

        # 实时预览区域
        ttk.Label(frame, text="配置预览", font=('Arial', 11, 'bold')).grid(row=row, column=0, columnspan=3, pady=(5, 10))
        row += 1

        preview_frame = ttk.LabelFrame(frame, text="当前配置", padding=10)
        preview_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), padx=5, pady=5)

        self.preview_text = tk.Text(preview_frame, height=6, width=70, wrap=tk.WORD,
                                   font=('Consolas', 10))
        self.preview_text.pack(fill=tk.BOTH, expand=True)

        # 更新预览
        threshold_var.trace('w', self.update_roi3_preview)
        enabled_var.trace('w', self.update_roi3_preview)
        self.update_roi3_preview()

        # 使用说明
        row += 1
        help_text = """ROI3覆盖逻辑说明:
1. 当ROI2检测为红色波峰时，系统会检查ROI3的峰值
2. 如果ROI3峰值 > 设定阈值，则将波峰覆盖为绿色
3. 这允许使用ROI3区域的高回声信号来"纠正"ROI2的分类
4. 适用于ROI2区域信号不稳定但ROI3区域信号更可靠的情况

建议设置:
- 100: 保守设置，更容易触发覆盖
- 115: 默认设置，平衡敏感性和特异性
- 130: 积极设置，需要更强的ROI3信号
- 150: 极高设置，只在强信号时触发"""

        help_frame = ttk.LabelFrame(frame, text="使用说明", padding=10)
        help_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), padx=5, pady=10)

        help_label = ttk.Label(help_frame, text=help_text, justify=tk.LEFT,
                              font=('Arial', 9))
        help_label.pack(fill=tk.BOTH, expand=True)

        frame.columnconfigure(1, weight=1)

    def set_roi3_threshold(self, value):
        """设置ROI3阈值"""
        self.roi3_override_vars["roi3_override.threshold"].set(str(value))

    def update_roi3_preview(self, *args):
        """更新ROI3配置预览"""
        try:
            enabled = self.roi3_override_vars["roi3_override.enabled"].get()
            threshold = self.roi3_override_vars["roi3_override.threshold"].get()
            require_data = self.roi3_override_vars["roi3_override.require_roi3_data"].get()

            if not enabled:
                preview = "ROI3覆盖功能: 已禁用\n\n系统将仅使用ROI2进行波峰分类，不会应用ROI3覆盖逻辑。"
            else:
                try:
                    threshold_val = float(threshold)
                    preview = f"""ROI3覆盖功能: 已启用

阈值设置: {threshold_val}
要求ROI3数据: {'是' if require_data else '否'}

工作逻辑:
- 当ROI2检测为红色波峰时
- 检查ROI3峰值是否 > {threshold_val}
- 如果是: 红色 → 绿色 (覆盖)
- 如果否: 保持红色不变"""
                except ValueError:
                    preview = "错误: 阈值设置无效，请输入有效数字"

            self.preview_text.delete(1.0, tk.END)
            self.preview_text.insert(1.0, preview)

        except Exception as e:
            pass

    def on_roi3_override_toggle(self):
        """ROI3覆盖开关切换时的处理"""
        enabled = self.roi3_override_vars["roi3_override.enabled"].get()
        # 可以在这里添加启用/禁用时需要执行的其他操作

    def browse_file(self, var):
        """浏览文件"""
        filename = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[
                ("视频文件", "*.mp4 *.avi *.mov *.mkv"),
                ("所有文件", "*.*")
            ]
        )
        if filename:
            var.set(filename)

    def load_config(self):
        """加载配置文件"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.config_data = json.load(f)
            else:
                self.config_data = {}
                messagebox.showwarning("警告", f"配置文件 {self.config_path} 不存在，将创建新配置")

            # 将配置数据加载到UI控件
            self.load_to_widgets()

            self.status_var.set(f"配置已加载: {self.config_path}")

        except Exception as e:
            messagebox.showerror("错误", f"加载配置文件失败: {str(e)}")
            self.status_var.set("加载失败")

    def load_to_widgets(self):
        """将配置数据加载到UI控件"""
        # 基础配置
        for key, var in self.basic_vars.items():
            value = self.get_nested_value(key, self.config_data)
            if isinstance(var, tk.BooleanVar):
                var.set(bool(value))
            else:
                var.set(str(value) if value is not None else "")

        # ROI配置
        for key, var in self.roi_vars.items():
            value = self.get_nested_value(key, self.config_data)
            if isinstance(var, tk.BooleanVar):
                var.set(bool(value))
            else:
                var.set(str(value) if value is not None else "")

        # 峰值检测配置
        for key, var in self.peak_vars.items():
            value = self.get_nested_value(key, self.config_data)
            if isinstance(var, tk.BooleanVar):
                var.set(bool(value))
            else:
                var.set(str(value) if value is not None else "")

        # ROI3覆盖配置
        for key, var in self.roi3_override_vars.items():
            value = self.get_nested_value(key, self.config_data)
            if isinstance(var, tk.BooleanVar):
                var.set(bool(value))
            else:
                var.set(str(value) if value is not None else "")

        # 确保ROI3扩展参数有默认值
        roi3_defaults = {
            "roi_capture.roi3_config.extension_params.left": "30",
            "roi_capture.roi3_config.extension_params.right": "40",
            "roi_capture.roi3_config.extension_params.top": "70",
            "roi_capture.roi3_config.extension_params.bottom": "30"
        }

        for key, default_value in roi3_defaults.items():
            if key in self.roi_vars and (not self.roi_vars[key].get() or self.roi_vars[key].get().strip() == ""):
                self.roi_vars[key].set(default_value)

    def get_nested_value(self, key_path: str, data: dict):
        """获取嵌套字典中的值"""
        keys = key_path.split('.')
        current = data
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        return current

    def set_nested_value(self, key_path: str, data: dict, value):
        """设置嵌套字典中的值"""
        keys = key_path.split('.')
        current = data
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value

    def save_config(self):
        """保存配置到文件"""
        try:
            # 从UI控件收集配置数据
            self.collect_from_widgets()

            # 保存到文件
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config_data, f, indent=2, ensure_ascii=False)

            self.status_var.set(f"配置已保存: {self.config_path}")
            messagebox.showinfo("成功", "配置已成功保存!")

        except Exception as e:
            messagebox.showerror("错误", f"保存配置文件失败: {str(e)}")
            self.status_var.set("保存失败")

    def save_as(self):
        """另存为配置文件"""
        try:
            filename = filedialog.asksaveasfilename(
                title="另存为配置文件",
                defaultextension=".json",
                filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")]
            )
            if filename:
                # 从UI控件收集配置数据
                self.collect_from_widgets()

                # 保存到指定文件
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(self.config_data, f, indent=2, ensure_ascii=False)

                self.status_var.set(f"配置已保存: {filename}")
                messagebox.showinfo("成功", f"配置已保存到: {filename}")

        except Exception as e:
            messagebox.showerror("错误", f"另存为失败: {str(e)}")
            self.status_var.set("另存为失败")

    def collect_from_widgets(self):
        """从UI控件收集配置数据"""
        # 基础配置
        for key, var in self.basic_vars.items():
            if isinstance(var, tk.BooleanVar):
                value = var.get()
            else:
                value = var.get()
                # 尝试转换为适当的类型
                if value:
                    try:
                        if '.' in value:
                            value = float(value)
                        else:
                            value = int(value)
                    except ValueError:
                        pass
            self.set_nested_value(key, self.config_data, value)

        # ROI配置
        for key, var in self.roi_vars.items():
            if isinstance(var, tk.BooleanVar):
                value = var.get()
            else:
                value = var.get()
                if value:
                    try:
                        value = int(value)
                    except ValueError:
                        pass
            self.set_nested_value(key, self.config_data, value)

        # 峰值检测配置
        for key, var in self.peak_vars.items():
            if isinstance(var, tk.BooleanVar):
                value = var.get()
            else:
                value = var.get()
                if value:
                    try:
                        if '.' in value:
                            value = float(value)
                        else:
                            value = int(value)
                    except ValueError:
                        pass
            self.set_nested_value(key, self.config_data, value)

        # ROI3覆盖配置
        for key, var in self.roi3_override_vars.items():
            if isinstance(var, tk.BooleanVar):
                value = var.get()
            else:
                value = var.get()
                if value:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
            self.set_nested_value(key, self.config_data, value)

    def import_roi1_image(self):
        """导入ROI1图片"""
        filename = filedialog.askopenfilename(
            title="选择ROI1图片",
            filetypes=[
                ("图片文件", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff"),
                ("所有文件", "*.*")
            ]
        )
        if filename:
            try:
                # 加载图片
                self.roi1_image = Image.open(filename)
                self.current_roi1_path = filename
                self.image_path_var.set(f"已加载: {os.path.basename(filename)}")

                # 检测图片序列
                print(f"[DEBUG] 开始检测图片序列: {filename}")
                self.detect_image_sequence(filename)
                print(f"[DEBUG] 序列检测完成，找到 {len(self.current_image_sequence)} 张图片")
                print(f"[DEBUG] 当前图片索引: {self.current_image_index}")

                # 更新可视化
                self.update_roi_visualization()

                # 连续检查：自动执行阈值处理
                if self.continuous_check_enabled.get():
                    self.root.after(200, self.auto_execute_threshold_processing)

                # 连续热力图：自动执行热力图处理（独立于连续检查）
                if self.continuous_heatmap_enabled.get():
                    self.root.after(300, self.auto_execute_heatmap_processing)

                # 显示序列信息
                if len(self.current_image_sequence) > 1:
                    sequence_info = f"序列: {len(self.current_image_sequence)}张图片 (按D/A或→/←切换)"
                    self.status_var.set(f"ROI1图片已导入: {os.path.basename(filename)} | {sequence_info}")
                    print(f"[INFO] {sequence_info}")
                else:
                    self.status_var.set(f"ROI1图片已导入: {os.path.basename(filename)}")
                    print(f"[INFO] 未检测到有效的图片序列")

            except Exception as e:
                messagebox.showerror("错误", f"导入图片失败: {str(e)}")
                self.status_var.set("图片导入失败")

    def clear_roi1_image(self):
        """清除ROI1图片"""
        self.roi1_image = None
        self.roi1_photo = None
        self.current_roi1_path = None
        self.image_path_var.set("未选择图片")
        self.roi_canvas.delete("all")
        self.pixel_info_var.set("请导入图片并在ROI1区域移动鼠标")  # 重置固定文本框
        self.status_var.set("ROI1图片已清除")

    def get_roi_config_values(self):
        """获取ROI配置值"""
        try:
            # ROI1配置
            roi1_x1 = int(self.roi_vars.get("roi_capture.default_config.x1", tk.StringVar()).get() or 0)
            roi1_y1 = int(self.roi_vars.get("roi_capture.default_config.y1", tk.StringVar()).get() or 0)
            roi1_x2 = int(self.roi_vars.get("roi_capture.default_config.x2", tk.StringVar()).get() or 100)
            roi1_y2 = int(self.roi_vars.get("roi_capture.default_config.y2", tk.StringVar()).get() or 100)

            # ROI2扩展参数
            roi2_left = int(self.roi_vars.get("roi_capture.roi2_config.extension_params.left", tk.StringVar()).get() or 20)
            roi2_right = int(self.roi_vars.get("roi_capture.roi2_config.extension_params.right", tk.StringVar()).get() or 30)
            roi2_top = int(self.roi_vars.get("roi_capture.roi2_config.extension_params.top", tk.StringVar()).get() or 60)
            roi2_bottom = int(self.roi_vars.get("roi_capture.roi2_config.extension_params.bottom", tk.StringVar()).get() or 20)

            # ROI3扩展参数
            roi3_left = int(self.roi_vars.get("roi_capture.roi3_config.extension_params.left", tk.StringVar()).get() or 30)
            roi3_right = int(self.roi_vars.get("roi_capture.roi3_config.extension_params.right", tk.StringVar()).get() or 40)
            roi3_top = int(self.roi_vars.get("roi_capture.roi3_config.extension_params.top", tk.StringVar()).get() or 70)
            roi3_bottom = int(self.roi_vars.get("roi_capture.roi3_config.extension_params.bottom", tk.StringVar()).get() or 30)

            return {
                'roi1': (roi1_x1, roi1_y1, roi1_x2, roi1_y2),
                'roi2': (roi2_left, roi2_right, roi2_top, roi2_bottom),
                'roi3': (roi3_left, roi3_right, roi3_top, roi3_bottom)
            }
        except (ValueError, AttributeError):
            # 如果配置无效，返回默认值
            return {
                'roi1': (0, 0, 100, 100),
                'roi2': (20, 30, 60, 20),
                'roi3': (30, 40, 70, 30)
            }

    def update_roi_visualization(self):
        """更新ROI可视化"""
        # 清除画布
        self.roi_canvas.delete("all")

        if self.roi1_image is None:
            # 如果没有图片，显示提示文本
            self.roi_canvas.create_text(
                320, 450,  # 居中在640x900画布中
                text="请导入ROI1图片以预览ROI区域\n\n支持格式：PNG, JPG, JPEG, BMP, TIFF",
                fill='gray',
                font=('Arial', 14),
                justify=tk.CENTER
            )
            return

        # 获取ROI配置
        roi_config = self.get_roi_config_values()

        # 获取画布尺寸
        canvas_width = self.roi_canvas.winfo_width()
        canvas_height = self.roi_canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            # 画布还未初始化，使用默认尺寸 (640x900)
            canvas_width = 640
            canvas_height = 900

        # 获取原始图片尺寸
        original_img_width, original_img_height = self.roi1_image.size

        # 获取ROI1配置并确保在图片范围内
        roi1_x1, roi1_y1, roi1_x2, roi1_y2 = roi_config['roi1']

        # 确保ROI1坐标在有效范围内
        roi1_x1 = max(0, min(roi1_x1, original_img_width - 1))
        roi1_y1 = max(0, min(roi1_y1, original_img_height - 1))
        roi1_x2 = max(roi1_x1 + 1, min(roi1_x2, original_img_width))
        roi1_y2 = max(roi1_y1 + 1, min(roi1_y2, original_img_height))

        # 确保ROI1有足够的最小尺寸，如果配置明显不匹配则使用图片的大部分区域
        roi1_width = roi1_x2 - roi1_x1
        roi1_height = roi1_y2 - roi1_y1
        min_size = 50

        # 如果ROI1区域相对于原图太小（小于原图的25%），则使用图片的大部分区域
        img_area = original_img_width * original_img_height
        roi1_area = roi1_width * roi1_height

        if roi1_area < img_area * 0.25:  # 如果ROI1面积小于原图面积的25%
            # 使用图片的80%作为ROI1区域，居中显示
            margin_x = int(original_img_width * 0.1)
            margin_y = int(original_img_height * 0.1)
            roi1_x1 = margin_x
            roi1_y1 = margin_y
            roi1_x2 = original_img_width - margin_x
            roi1_y2 = original_img_height - margin_y
            roi1_width = roi1_x2 - roi1_x1
            roi1_height = roi1_y2 - roi1_y1
        else:
            # 否则应用最小尺寸逻辑
            if roi1_width < min_size:
                # 居中扩展ROI1宽度
                center_x = (roi1_x1 + roi1_x2) // 2
                roi1_x1 = max(0, center_x - min_size // 2)
                roi1_x2 = min(original_img_width, roi1_x1 + min_size)
                roi1_width = roi1_x2 - roi1_x1

            if roi1_height < min_size:
                # 居中扩展ROI1高度
                center_y = (roi1_y1 + roi1_y2) // 2
                roi1_y1 = max(0, center_y - min_size // 2)
                roi1_y2 = min(original_img_height, roi1_y1 + min_size)
                roi1_height = roi1_y2 - roi1_y1

        # 从原始图片中提取ROI1区域作为背景
        roi1_region = self.roi1_image.crop((roi1_x1, roi1_y1, roi1_x2, roi1_y2))

        # 计算缩放比例以适应画布
        scale_x = (canvas_width - 20) / roi1_width
        scale_y = (canvas_height - 20) / roi1_height
        scale = min(scale_x, scale_y, 1.0)  # 不放大图片

        # 缩放ROI1图片
        display_width = int(roi1_width * scale)
        display_height = int(roi1_height * scale)
        display_image = roi1_region.resize((display_width, display_height), Image.Resampling.LANCZOS)

        # 转换为PhotoImage并显示
        self.roi1_photo = ImageTk.PhotoImage(display_image)

        # 居中显示ROI1图片作为背景
        x_offset = (canvas_width - display_width) // 2
        y_offset = (canvas_height - display_height) // 2

        self.roi_canvas.create_image(
            canvas_width // 2, canvas_height // 2,
            image=self.roi1_photo,
            anchor=tk.CENTER
        )

        # ROI1中心点作为交点（在ROI1坐标系中的坐标）
        center_x = roi1_width // 2
        center_y = roi1_height // 2

        # ROI2区域（基于ROI1中心点）
        roi2_left = center_x - roi_config['roi2'][0]
        roi2_top = center_y - roi_config['roi2'][2]
        roi2_right = center_x + roi_config['roi2'][1]
        roi2_bottom = center_y + roi_config['roi2'][3]

        # ROI3区域（基于ROI1中心点，更大范围）
        roi3_left = center_x - roi_config['roi3'][0]
        roi3_top = center_y - roi_config['roi3'][2]
        roi3_right = center_x + roi_config['roi3'][1]
        roi3_bottom = center_y + roi_config['roi3'][3]

        # 转换ROI2、ROI3坐标到显示坐标系
        roi2_img_left = (roi2_left * scale) + x_offset
        roi2_img_top = (roi2_top * scale) + y_offset
        roi2_img_right = (roi2_right * scale) + x_offset
        roi2_img_bottom = (roi2_bottom * scale) + y_offset

        roi3_img_left = (roi3_left * scale) + x_offset
        roi3_img_top = (roi3_top * scale) + y_offset
        roi3_img_right = (roi3_right * scale) + x_offset
        roi3_img_bottom = (roi3_bottom * scale) + y_offset

        # 交点坐标（ROI1中心点在显示坐标系中的位置）
        intersection_x = (center_x * scale) + x_offset
        intersection_y = (center_y * scale) + y_offset

        # ROI1边界（显示坐标，即背景图片边界）
        roi1_img_left = x_offset
        roi1_img_top = y_offset
        roi1_img_right = x_offset + display_width
        roi1_img_bottom = y_offset + display_height

        # 绘制ROI3区域（蓝色虚线，叠加在背景上）
        self.roi_canvas.create_rectangle(
            roi3_img_left, roi3_img_top, roi3_img_right, roi3_img_bottom,
            outline='blue', width=2, dash=(8, 4)
        )

        # 绘制ROI2区域（红色实线，叠加在背景上）
        self.roi_canvas.create_rectangle(
            roi2_img_left, roi2_img_top, roi2_img_right, roi2_img_bottom,
            outline='red', width=3
        )

        # 绘制交点（ROI1中心点，绿色圆点，叠加在背景上）
        self.roi_canvas.create_oval(
            intersection_x - 6, intersection_y - 6,
            intersection_x + 6, intersection_y + 6,
            fill='lime', outline='darkgreen', width=2
        )

        # 添加标签
        self.roi_canvas.create_text(
            roi2_img_left - 5, roi2_img_top - 5,
            text="ROI2", fill='red', font=('Arial', 10, 'bold'), anchor='se'
        )

        self.roi_canvas.create_text(
            roi3_img_left - 5, roi3_img_top - 5,
            text="ROI3", fill='blue', font=('Arial', 10, 'bold'), anchor='se'
        )

        # 添加ROI1标签
        self.roi_canvas.create_text(
            roi1_img_left + 5, roi1_img_top + 5,
            text="ROI1(背景)", fill='darkgreen', font=('Arial', 10, 'bold'), anchor='nw'
        )

        # 添加尺寸信息
        roi2_width = roi2_right - roi2_left
        roi2_height = roi2_bottom - roi2_top
        roi3_width = roi3_right - roi3_left
        roi3_height = roi3_bottom - roi3_top

        # 在ROI2区域右下角显示尺寸
        self.roi_canvas.create_text(
            roi2_img_right - 5, roi2_img_bottom - 5,
            text=f"{roi2_width}x{roi2_height}",
            fill='red', font=('Arial', 8), anchor='se'
        )

        # 在ROI3区域右下角显示尺寸
        self.roi_canvas.create_text(
            roi3_img_right - 5, roi3_img_bottom - 5,
            text=f"{roi3_width}x{roi3_height}",
            fill='blue', font=('Arial', 8), anchor='se'
        )

        # 在ROI1区域左下角显示图片信息
        self.roi_canvas.create_text(
            roi1_img_left + 5, roi1_img_bottom - 5,
            text=f"原图: {original_img_width}x{original_img_height} | ROI1: {roi1_width}x{roi1_height} | 缩放: {scale:.2f}x",
            fill='darkgreen', font=('Arial', 8), anchor='sw'
        )

        # 绘制ROI2和ROI3的灰度曲线图
        self.draw_grayscale_curves(roi_config, original_img_width, original_img_height)

    def compute_roi_grayscale(self, roi1_image, roi_x, roi_y, roi_width, roi_height):
        """计算ROI区域的平均灰度值"""
        try:
            # 裁剪ROI区域
            roi_region = roi1_image.crop((roi_x, roi_y, roi_x + roi_width, roi_y + roi_height))

            # 转换为灰度图像
            if roi_region.mode != 'L':
                roi_region = roi_region.convert('L')

            # 计算平均灰度值
            import numpy as np
            roi_array = np.array(roi_region)
            avg_gray = float(np.mean(roi_array))

            return avg_gray
        except Exception as e:
            print(f"计算ROI灰度值失败: {e}")
            return 0.0

    def draw_grayscale_curves(self, roi_config, img_width, img_height):
        """绘制ROI2和ROI3的灰度直方图（像素分布曲线）"""
        # 清除曲线画布
        self.curve_canvas.delete("all")

        if self.roi1_image is None:
            self.curve_canvas.create_text(
                300, 450,  # 居中在600x900画布中
                text="请导入ROI1图片以显示灰度直方图\n\nX轴：灰度值(0-255)\nY轴：像素数量",
                fill='gray',
                font=('Arial', 14),
                justify=tk.CENTER
            )
            return

        try:
            # 获取画布尺寸
            canvas_width = self.curve_canvas.winfo_width()
            canvas_height = self.curve_canvas.winfo_height()

            if canvas_width <= 1 or canvas_height <= 1:
                canvas_width = 600
                canvas_height = 900

            # 设置绘图区域（为600x900画布优化边距）
            margin_left = 80
            margin_right = 40
            margin_top = 40
            margin_bottom = 100

            plot_width = canvas_width - margin_left - margin_right
            plot_height = canvas_height - margin_top - margin_bottom

            # 绘制坐标轴
            # X轴（灰度值）
            self.curve_canvas.create_line(
                margin_left, margin_top + plot_height,
                margin_left + plot_width, margin_top + plot_height,
                fill='black', width=2
            )
            # Y轴（像素个数）
            self.curve_canvas.create_line(
                margin_left, margin_top,
                margin_left, margin_top + plot_height,
                fill='black', width=2
            )

            # 添加坐标轴标签
            self.curve_canvas.create_text(
                margin_left + plot_width // 2, canvas_height - 15,
                text="灰度值 (0-255)", fill='black', font=('Arial', 12, 'bold')
            )
            self.curve_canvas.create_text(
                25, margin_top + plot_height // 2,
                text="像素个数", fill='black', font=('Arial', 12, 'bold'), angle=90
            )

            # 获取ROI1实际裁剪区域（与update_roi_visualization保持一致）
            roi1_x1, roi1_y1, roi1_x2, roi1_y2 = roi_config['roi1']
            roi1_width = roi1_x2 - roi1_x1
            roi1_height = roi1_y2 - roi1_y1

            # 获取原始图片尺寸
            original_img_width, original_img_height = img_width, img_height

            # 确保ROI1坐标在有效范围内
            roi1_x1 = max(0, min(roi1_x1, original_img_width - 1))
            roi1_y1 = max(0, min(roi1_y1, original_img_height - 1))
            roi1_x2 = max(roi1_x1 + 1, min(roi1_x2, original_img_width))
            roi1_y2 = max(roi1_y1 + 1, min(roi1_y2, original_img_height))

            # 重新计算ROI1实际尺寸
            roi1_width = roi1_x2 - roi1_x1
            roi1_height = roi1_y2 - roi1_y1

            # 如果ROI1区域相对于原图太小，则使用图片的大部分区域（与update_roi_visualization保持一致）
            min_size = 50
            img_area = original_img_width * original_img_height
            roi1_area = roi1_width * roi1_height

            if roi1_area < img_area * 0.25:  # 如果ROI1面积小于原图面积的25%
                # 使用图片的80%作为ROI1区域，居中显示
                margin_x = int(original_img_width * 0.1)
                margin_y = int(original_img_height * 0.1)
                roi1_x1 = margin_x
                roi1_y1 = margin_y
                roi1_x2 = original_img_width - margin_x
                roi1_y2 = original_img_height - margin_y
                roi1_width = roi1_x2 - roi1_x1
                roi1_height = roi1_y2 - roi1_y1
            else:
                # 否则应用最小尺寸逻辑
                if roi1_width < min_size:
                    # 居中扩展ROI1宽度
                    center_x_abs = (roi1_x1 + roi1_x2) // 2
                    roi1_x1 = max(0, center_x_abs - min_size // 2)
                    roi1_x2 = min(original_img_width, roi1_x1 + min_size)
                    roi1_width = roi1_x2 - roi1_x1

                if roi1_height < min_size:
                    # 居中扩展ROI1高度
                    center_y_abs = (roi1_y1 + roi1_y2) // 2
                    roi1_y1 = max(0, center_y_abs - min_size // 2)
                    roi1_y2 = min(original_img_height, roi1_y1 + min_size)
                    roi1_height = roi1_y2 - roi1_y1

            # ROI1中心点（相对于ROI1裁剪区域）
            center_x = roi1_width // 2
            center_y = roi1_height // 2

            # ROI2和ROI3的扩展参数
            roi2_left, roi2_right, roi2_top, roi2_bottom = roi_config['roi2']
            roi3_left, roi3_right, roi3_top, roi3_bottom = roi_config['roi3']

            # 计算ROI2和ROI3相对于ROI1裁剪区域的坐标
            roi2_x = center_x - roi2_left
            roi2_y = center_y - roi2_top
            roi2_width = roi2_left + roi2_right
            roi2_height = roi2_top + roi2_bottom

            roi3_x = center_x - roi3_left
            roi3_y = center_y - roi3_top
            roi3_width = roi3_left + roi3_right
            roi3_height = roi3_top + roi3_bottom

            # 创建ROI1裁剪区域用于直方图计算
            roi1_region = self.roi1_image.crop((roi1_x1, roi1_y1, roi1_x2, roi1_y2))

            # 获取ROI2和ROI3的灰度直方图数据（使用ROI1裁剪区域）
            roi2_histogram = self.compute_grayscale_histogram(
                roi1_region, roi2_x, roi2_y, roi2_width, roi2_height
            )
            roi3_histogram = self.compute_grayscale_histogram(
                roi1_region, roi3_x, roi3_y, roi3_width, roi3_height
            )

            # 调试输出
            print(f"[DEBUG] ROI2区域: x={roi2_x}, y={roi2_y}, w={roi2_width}, h={roi2_height}")
            print(f"[DEBUG] ROI3区域: x={roi3_x}, y={roi3_y}, w={roi3_width}, h={roi3_height}")
            print(f"[DEBUG] ROI2直方图长度: {len(roi2_histogram) if roi2_histogram else 0}")
            print(f"[DEBUG] ROI3直方图长度: {len(roi3_histogram) if roi3_histogram else 0}")
            if roi2_histogram:
                roi2_total = sum(roi2_histogram)
                roi2_max = max(roi2_histogram)
                print(f"[DEBUG] ROI2总像素: {roi2_total}, 最大像素数: {roi2_max}")
            if roi3_histogram:
                roi3_total = sum(roi3_histogram)
                roi3_max = max(roi3_histogram)
                print(f"[DEBUG] ROI3总像素: {roi3_total}, 最大像素数: {roi3_max}")

            # 确定Y轴范围（像素个数）- 强制使用500作为默认最大值
            roi2_max_count = max(roi2_histogram) if roi2_histogram else 1
            roi3_max_count = max(roi3_histogram) if roi3_histogram else 1
            base_max_count = 500  # 固定使用500作为Y轴默认最大值，忽略实际ROI像素数

            print(f"[INFO] 强制设置Y轴默认范围为0-{base_max_count}像素")
            print(f"[DEBUG] ROI2实际最大像素数: {roi2_max_count} (可能被截断)")
            print(f"[DEBUG] ROI3实际最大像素数: {roi3_max_count} (可能被截断)")

            # 应用Y轴缩放因子
            max_count = int(base_max_count / self.y_zoom_factor)  # 缩放因子越大，显示的Y轴范围越小（放大）

            print(f"[DEBUG] ROI2最大像素数: {roi2_max_count}")
            print(f"[DEBUG] ROI3最大像素数: {roi3_max_count}")
            print(f"[DEBUG] 基础最大值: {base_max_count}")
            print(f"[DEBUG] Y轴缩放因子: {self.y_zoom_factor}")
            print(f"[DEBUG] 最终Y轴最大值: {max_count}")

            # 绘制ROI2灰度直方图（红色）
            roi2_points = []
            for gray_val in range(256):
                if gray_val < len(roi2_histogram):
                    count = roi2_histogram[gray_val]
                    canvas_x = margin_left + (gray_val * plot_width // 255)
                    # 限制count不超过max_count，防止曲线超出Y轴范围
                    display_count = min(count, max_count)
                    canvas_y = margin_top + plot_height - (display_count * plot_height // max_count)
                    roi2_points.extend([canvas_x, canvas_y])

            # 使用平滑线绘制直方图
            if len(roi2_points) >= 4:
                self.curve_canvas.create_line(
                    roi2_points, fill='red', width=2, smooth=False
                )

            # 绘制ROI3灰度直方图（蓝色）
            roi3_points = []
            for gray_val in range(256):
                if gray_val < len(roi3_histogram):
                    count = roi3_histogram[gray_val]
                    canvas_x = margin_left + (gray_val * plot_width // 255)
                    # 限制count不超过max_count，防止曲线超出Y轴范围
                    display_count = min(count, max_count)
                    canvas_y = margin_top + plot_height - (display_count * plot_height // max_count)
                    roi3_points.extend([canvas_x, canvas_y])

            if len(roi3_points) >= 4:
                self.curve_canvas.create_line(
                    roi3_points, fill='blue', width=2, smooth=False
                )

            # 添加X轴刻度（灰度值）
            x_ticks = [0, 64, 128, 192, 255]
            for gray_val in x_ticks:
                canvas_x = margin_left + (gray_val * plot_width // 255)
                self.curve_canvas.create_line(
                    canvas_x, margin_top + plot_height,
                    canvas_x, margin_top + plot_height + 8,
                    fill='black', width=1
                )
                self.curve_canvas.create_text(
                    canvas_x, margin_top + plot_height + 20,
                    text=str(gray_val), fill='black', font=('Arial', 10, 'bold')
                )

            # 添加Y轴刻度（像素个数）
            y_ticks = [0, max_count//4, max_count//2, max_count*3//4, max_count]
            for i, count in enumerate(y_ticks):
                canvas_y = margin_top + plot_height - (count * plot_height // max_count)
                self.curve_canvas.create_line(
                    margin_left - 5, canvas_y, margin_left, canvas_y,
                    fill='black', width=1
                )
                self.curve_canvas.create_text(
                    margin_left - 15, canvas_y,
                    text=f"{count}", fill='black', font=('Arial', 10, 'bold'),
                    anchor='e'
                )

            # 注释：灰度直方图不需要显示峰值检测阈值线
            # 阈值线适用于信号处理图表，不适用于像素分布直方图

            # 添加统计信息和缩放级别
            roi2_total = sum(roi2_histogram) if roi2_histogram else 0
            roi3_total = sum(roi3_histogram) if roi3_histogram else 0
            roi2_avg = roi2_total / 256 if roi2_total > 0 else 0
            roi3_avg = roi3_total / 256 if roi3_total > 0 else 0

            # 获取缩放信息
            zoom_info = self.get_y_zoom_info()
            zoom_text = f"Y轴缩放: {zoom_info['zoom_percentage']}%" if zoom_info['is_zoomed'] else "Y轴缩放: 100%"

            stats_text = f"ROI2: 总像素{roi2_total} 平均{roi2_avg:.1f}  |  ROI3: 总像素{roi3_total} 平均{roi3_avg:.1f}  |  {zoom_text}"
            self.curve_canvas.create_text(
                canvas_width // 2, 20,
                text=stats_text, fill='black', font=('Arial', 11, 'bold')
            )

            # 添加缩放操作提示
            if self.roi1_image:  # 只在有图片时显示提示
                hint_text = "提示: 使用鼠标滚轮可以缩放Y轴范围 | Ctrl+滚轮重置缩放"
                self.curve_canvas.create_text(
                    canvas_width // 2, 40,
                    text=hint_text, fill='gray', font=('Arial', 9)
                )

            # 添加ROI尺寸信息
            size_text = f"ROI2: {roi2_width}x{roi2_height}={roi2_width*roi2_height}像素  |  ROI3: {roi3_width}x{roi3_height}={roi3_width*roi3_height}像素"
            self.curve_canvas.create_text(
                canvas_width // 2, canvas_height - 35,
                text=size_text, fill='gray', font=('Arial', 9)
            )

        except Exception as e:
            self.curve_canvas.create_text(
                canvas_width // 2, canvas_height // 2,
                text=f"绘制灰度直方图失败: {str(e)}",
                fill='red', font=('Arial', 10)
            )

    def compute_grayscale_histogram(self, roi1_image, roi_x, roi_y, roi_width, roi_height):
        """计算ROI区域的灰度直方图（0-255灰度值的像素分布）"""
        try:
            # 确保ROI坐标在图片范围内
            img_width, img_height = roi1_image.size
            roi_x = max(0, roi_x)
            roi_y = max(0, roi_y)
            roi_x2 = min(img_width, roi_x + roi_width)
            roi_y2 = min(img_height, roi_y + roi_height)

            actual_width = roi_x2 - roi_x
            actual_height = roi_y2 - roi_y

            if actual_width <= 0 or actual_height <= 0:
                return []

            # 裁剪ROI区域
            roi_region = roi1_image.crop((roi_x, roi_y, roi_x2, roi_y2))

            # 转换为灰度图像
            if roi_region.mode != 'L':
                roi_region = roi_region.convert('L')

            # 转换为numpy数组
            import numpy as np
            roi_array = np.array(roi_region)

            # 计算0-255每个灰度值的像素个数
            histogram = [0] * 256
            for gray_val in range(256):
                histogram[gray_val] = int(np.sum(roi_array == gray_val))

            return histogram

        except Exception as e:
            print(f"计算灰度直方图失败: {e}")
            return []

    def on_curve_canvas_mousewheel(self, event):
        """处理曲线画布的鼠标滚轮事件，用于Y轴缩放"""
        try:
            # 获取滚轮滚动方向
            if event.delta:  # Windows
                delta = event.delta / 120  # 标准化为-1或1
            elif event.num == 4:  # Linux 向上滚动
                delta = 1
            elif event.num == 5:  # Linux 向下滚动
                delta = -1
            else:
                return

            # 计算新的缩放因子
            if delta > 0:  # 向上滚动，放大
                new_zoom = self.y_zoom_factor * (1 + self.y_zoom_step)
            else:  # 向下滚动，缩小
                new_zoom = self.y_zoom_factor * (1 - self.y_zoom_step)

            # 限制缩放范围
            new_zoom = max(self.y_min_zoom, min(self.y_max_zoom, new_zoom))

            # 如果缩放因子发生变化，更新显示
            if new_zoom != self.y_zoom_factor:
                self.y_zoom_factor = new_zoom
                print(f"[INFO] Y轴缩放: {self.y_zoom_factor:.2f}x")

                # 重新绘制曲线（如果有ROI图片）
                if self.roi1_image:
                    roi_config = self.get_roi_config_values()
                    original_img_width, original_img_height = self.roi1_image.size
                    self.draw_grayscale_curves(roi_config, original_img_width, original_img_height)

        except Exception as e:
            print(f"[ERROR] 鼠标滚轮事件处理失败: {e}")

    def on_curve_canvas_mousewheel_reset(self, event):
        """重置Y轴缩放为默认值"""
        try:
            if self.y_zoom_factor != 1.0:
                self.y_zoom_factor = 1.0
                print(f"[INFO] Y轴缩放已重置: {self.y_zoom_factor:.2f}x")

                # 重新绘制曲线（如果有ROI图片）
                if self.roi1_image:
                    roi_config = self.get_roi_config_values()
                    original_img_width, original_img_height = self.roi1_image.size
                    self.draw_grayscale_curves(roi_config, original_img_width, original_img_height)

        except Exception as e:
            print(f"[ERROR] 重置Y轴缩放失败: {e}")

    def on_roi_canvas_mouse_motion(self, event):
        """处理ROI1画布的鼠标移动事件，显示当前位置的灰度值"""
        try:
            # 确保有ROI1图片
            if not self.roi1_image:
                no_image_msg = "请先导入图片"
                self.pixel_info_var.set(no_image_msg)
                self.status_var.set(no_image_msg)
                return

            # 获取画布尺寸
            canvas_width = self.roi_canvas.winfo_width()
            canvas_height = self.roi_canvas.winfo_height()

            # 如果画布还没有渲染完成，返回
            if canvas_width <= 1 or canvas_height <= 1:
                return

            # 获取原始图片尺寸
            img_width, img_height = self.roi1_image.size

            # 计算图片在画布中的显示区域（考虑缩放和居中）
            scale_x = canvas_width / img_width
            scale_y = canvas_height / img_height
            scale = min(scale_x, scale_y)  # 保持宽高比

            # 计算图片在画布中的实际显示区域
            display_width = img_width * scale
            display_height = img_height * scale
            offset_x = (canvas_width - display_width) / 2
            offset_y = (canvas_height - display_height) / 2

            # 检查鼠标是否在图片区域内
            if (event.x >= offset_x and event.x < offset_x + display_width and
                event.y >= offset_y and event.y < offset_y + display_height):

                # 将画布坐标转换为图片坐标
                img_x = int((event.x - offset_x) / scale)
                img_y = int((event.y - offset_y) / scale)

                # 确保坐标在图片范围内
                img_x = max(0, min(img_x, img_width - 1))
                img_y = max(0, min(img_y, img_height - 1))

                # 获取该位置的灰度值
                if self.roi1_image.mode == 'RGB':
                    # 如果是RGB图片，转换为灰度值
                    r, g, b = self.roi1_image.getpixel((img_x, img_y))
                    # 使用标准灰度转换公式
                    gray_value = int(0.299 * r + 0.587 * g + 0.114 * b)
                    pixel_info = f"坐标: ({img_x:4d}, {img_y:4d}) | RGB: ({r:3d}, {g:3d}, {b:3d}) | 灰度: {gray_value:3d}"
                elif self.roi1_image.mode == 'L':
                    # 如果已经是灰度图
                    gray_value = self.roi1_image.getpixel((img_x, img_y))
                    pixel_info = f"坐标: ({img_x:4d}, {img_y:4d}) | 灰度: {gray_value:3d}"
                else:
                    # 其他模式，转换为RGB再处理
                    rgb_image = self.roi1_image.convert('RGB')
                    r, g, b = rgb_image.getpixel((img_x, img_y))
                    gray_value = int(0.299 * r + 0.587 * g + 0.114 * b)
                    pixel_info = f"坐标: ({img_x:4d}, {img_y:4d}) | 灰度: {gray_value:3d}"

                # 更新固定文本框显示
                self.pixel_info_var.set(pixel_info)
                # 同时更新状态栏
                self.status_var.set(f"像素信息: {pixel_info}")

            else:
                # 鼠标不在图片区域内，显示提示信息
                canvas_pos_info = f"鼠标位置: ({event.x:3d}, {event.y:3d}) - 在图片区域外"
                self.pixel_info_var.set(canvas_pos_info)
                # 同时更新状态栏
                self.status_var.set(f"画布坐标: ({event.x:3d}, {event.y:3d}) - 在图片区域外")

        except Exception as e:
            print(f"[ERROR] 鼠标移动事件处理失败: {e}")
            error_msg = "获取像素信息失败"
            self.pixel_info_var.set(error_msg)
            self.status_var.set(error_msg)

    def on_roi_canvas_mouse_leave(self, event):
        """处理ROI1画布的鼠标离开事件，重置灰度值显示"""
        try:
            if self.roi1_image:
                leave_msg = "鼠标已离开ROI1区域"
                self.pixel_info_var.set(leave_msg)
                self.status_var.set(leave_msg)
            else:
                no_image_msg = "请导入图片并在ROI1区域移动鼠标"
                self.pixel_info_var.set(no_image_msg)
                self.status_var.set(no_image_msg)
        except Exception as e:
            print(f"[ERROR] 鼠标离开事件处理失败: {e}")

    def on_threshold_submit(self):
        """处理阈值提取提交"""
        try:
            if not self.roi1_image:
                self.status_var.set("请先导入ROI1图片")
                return

            # 验证阈值输入
            lower = self.threshold_lower_var.get()
            upper = self.threshold_upper_var.get()

            if lower < 0 or upper > 255 or lower > upper:
                self.status_var.set("阈值范围无效 (0-255, 且下限≤上限)")
                return

            # 如果当前处于热力图模式，清除热力图（互斥显示）
            if hasattr(self, 'heatmap_mode') and self.heatmap_mode:
                self.heat_map = None
                self.heatmap_mode = False

            # 提取ROI3
            roi_config = self.get_roi_config_values()
            roi3_image, roi3_coords = self.extract_roi3_from_roi1(self.roi1_image, roi_config)

            if roi3_image is None:
                self.status_var.set("ROI3提取失败")
                return

            # 应用阈值
            self.threshold_mask = self.apply_threshold_extraction(roi3_image, lower, upper)

            if self.threshold_mask is None:
                self.status_var.set("阈值处理失败")
                return

            # 创建叠加
            self.current_roi3_coords = roi3_coords
            self.current_mask_for_overlay = self.threshold_mask
            self.update_overlay()

            # 计算并显示统计信息
            self.calculate_mask_statistics()

            self.status_var.set(f"阈值提取完成: {lower}-{upper}")

        except Exception as e:
            self.status_var.set(f"阈值提取失败: {str(e)}")

    def on_threshold_clear(self):
        """处理阈值清除"""
        self.threshold_mask = None
        self.largest_component_mask = None
        self.current_overlay_image = None
        self.current_overlay_photo = None
        self.current_roi3_coords = None
        self.current_mask_for_overlay = None

        # 重置统计信息
        self.total_pixels_var.set("0")
        self.largest_component_pixels_var.set("0")
        self.component_count_var.set("0")

        # 更新画布（不显示叠加）
        self.update_roi_visualization()

        self.status_var.set("阈值已清除")

    def on_max_component(self):
        """处理最大连通域提取"""
        try:
            if self.threshold_mask is None:
                self.status_var.set("请先进行阈值提取")
                return

            # 如果当前处于热力图模式，清除热力图（互斥显示）
            if hasattr(self, 'heatmap_mode') and self.heatmap_mode:
                self.heat_map = None
                self.heatmap_mode = False

            # 执行连通域分析
            largest_mask, largest_area, component_count = self.analyze_connected_components(self.threshold_mask)

            if largest_mask is None:
                self.status_var.set("未找到连通域")
                return

            self.largest_component_mask = largest_mask

            # 使用最大连通域作为当前mask
            self.current_mask_for_overlay = largest_mask

            # 更新叠加
            self.update_overlay()

            # 更新统计信息
            self.largest_component_pixels_var.set(str(largest_area))
            self.component_count_var.set(str(component_count))

            self.status_var.set(f"最大连通域提取完成: {largest_area}像素")

        except Exception as e:
            self.status_var.set(f"连通域分析失败: {str(e)}")

    def auto_execute_threshold_processing(self):
        """自动执行阈值提取和最大连通域处理"""
        try:
            if not self.continuous_check_enabled.get():
                return False

            if not self.roi1_image:
                return False

            print("[INFO] 连续检查模式：自动执行阈值提取...")

            # 执行阈值提取
            self.on_threshold_submit()

            # 延迟执行最大连通域分析
            self.root.after(100, self._auto_execute_max_component)

            # 延迟执行热力图显示（如果启用连续热力图）
            if self.continuous_heatmap_enabled.get():
                self.root.after(200, self._auto_execute_heat_map)

            return True

        except Exception as e:
            print(f"[ERROR] 连续检查执行失败: {e}")
            self.status_var.set(f"连续检查失败: {str(e)}")
            return False

    def auto_execute_heatmap_processing(self):
        """自动执行热力图处理"""
        try:
            if not self.continuous_heatmap_enabled.get():
                return False

            if not self.roi1_image:
                return False

            print("[INFO] 连续热力图模式：自动执行热力图显示...")

            # 延迟执行热力图显示
            self.root.after(100, self._auto_execute_heat_map)

            return True

        except Exception as e:
            print(f"[ERROR] 连续热力图执行失败: {e}")
            self.status_var.set(f"连续热力图失败: {str(e)}")
            return False

    def _auto_execute_max_component(self):
        """延迟执行最大连通域分析"""
        try:
            if self.threshold_mask is not None:
                print("[INFO] 连续检查模式：自动执行最大连通域分析...")
                self.on_max_component()
                self.status_var.set("连续检查：自动完成提取+最大连通域分析")

        except Exception as e:
            print(f"[ERROR] 连续检查-最大连通域执行失败: {e}")

    def _auto_execute_heat_map(self):
        """延迟执行热力图显示"""
        try:
            # 检查是否需要执行热力图显示
            should_execute = False
            status_message = ""

            if self.continuous_check_enabled.get() and self.roi1_image:
                # 连续检查模式下的热力图执行
                should_execute = True
                status_message = "连续检查：自动完成提取+最大连通域+热力图显示"
                print("[INFO] 连续检查模式：自动执行热力图显示...")
            elif self.continuous_heatmap_enabled.get() and self.roi1_image:
                # 独立的连续热力图模式
                should_execute = True
                status_message = "连续热力图：自动完成热力图显示"
                print("[INFO] 连续热力图模式：自动执行热力图显示...")

            if should_execute:
                self.on_heatmap_submit()
                self.status_var.set(status_message)

        except Exception as e:
            print(f"[ERROR] 热力图执行失败: {e}")

    def on_statistics(self):
        """处理统计计算"""
        self.calculate_mask_statistics()

    def on_overlay_toggle(self):
        """处理叠加显示切换"""
        self.update_roi_canvas_with_overlay()

    def on_alpha_change(self, value):
        """处理透明度变化"""
        try:
            alpha = float(value)
            self.alpha_value_label.config(text=f"{alpha:.2f}")
            self.update_overlay()
        except:
            pass

    def on_heatmap_submit(self):
        """处理热力图显示按钮点击"""
        try:
            if not self.roi1_image:
                messagebox.showwarning("警告", "请先导入ROI1图像")
                return

            self.status_var.set("正在生成热力图...")
            self.root.update()

            # 获取ROI配置
            roi_config = self.get_roi_config_values()
            if not roi_config.get('roi3'):
                messagebox.showwarning("警告", "请先配置ROI3扩展参数")
                return

            # 提取ROI3图像
            roi3_image, roi3_coords = self.extract_roi3_from_roi1(self.roi1_image, roi_config)
            if roi3_image is None:
                messagebox.showerror("错误", "ROI3提取失败")
                return

            # 生成热力图
            print("[DEBUG] 开始生成热力图...")
            self.heat_map = self.apply_heat_map_extraction(roi3_image)
            if self.heat_map is None:
                print("[ERROR] 热力图生成失败")
                messagebox.showerror("错误", "热力图生成失败")
                return

            print(f"[DEBUG] 热力图生成成功: {self.heat_map.shape}")

            # 设置热力图模式
            self.heatmap_mode = True

            # 保存ROI3坐标（用于叠加）
            self.current_roi3_coords = roi3_coords
            print(f"[DEBUG] ROI3坐标已保存: {roi3_coords}")

            # 禁用阈值叠加（互斥显示）
            self.overlay_enabled.set(False)

            # 更新热力图显示
            print("[DEBUG] 开始更新热力图叠加...")
            self.update_heat_map_overlay()
            print("[DEBUG] 热力图叠加更新完成")

            # 更新统计信息
            self.calculate_heat_map_statistics()

            self.status_var.set("热力图显示完成")

        except Exception as e:
            messagebox.showerror("错误", f"热力图处理失败: {e}")

    def on_heatmap_clear(self):
        """处理清除热力图按钮点击"""
        try:
            # 清除热力图数据
            self.heat_map = None
            self.heatmap_mode = False

            # 更新画布显示
            self.update_roi_visualization()

            # 清除统计信息
            self.total_pixels_var.set("0")
            self.largest_pixels_var.set("0")
            self.component_count_var.set("0")

            self.status_var.set("热力图已清除")

        except Exception as e:
            print(f"[ERROR] 清除热力图失败: {e}")

    def on_heatmap_alpha_change(self, value):
        """处理热力图透明度变化"""
        try:
            alpha = float(value)
            self.heatmap_alpha_value_label.config(text=f"{alpha:.2f}")
            if self.heatmap_mode and self.heat_map is not None:
                self.update_heat_map_overlay()
        except:
            pass

    def calculate_heat_map_statistics(self):
        """计算热力图统计信息"""
        try:
            if self.heat_map is not None:
                import numpy as np
                # 统计非零像素（实际上所有像素都有颜色）
                total_pixels = self.heat_map.shape[0] * self.heat_map.shape[1]
                self.total_pixels_var.set(str(total_pixels))

                # 对于热力图，最大连通域就是整个热力图区域
                # 使用getattr避免属性不存在的错误
                if hasattr(self, 'largest_pixels_var'):
                    self.largest_pixels_var.set(str(total_pixels))
                if hasattr(self, 'component_count_var'):
                    self.component_count_var.set("1")
            else:
                self.total_pixels_var.set("0")
                if hasattr(self, 'largest_pixels_var'):
                    self.largest_pixels_var.set("0")
                if hasattr(self, 'component_count_var'):
                    self.component_count_var.set("0")

        except Exception as e:
            print(f"[ERROR] 热力图统计计算失败: {e}")

    def calculate_mask_statistics(self):
        """计算mask统计信息"""
        try:
            if hasattr(self, 'current_mask_for_overlay') and self.current_mask_for_overlay is not None:
                import numpy as np
                mask = self.current_mask_for_overlay
                total_pixels = np.sum(mask > 0)
                self.total_pixels_var.set(str(total_pixels))
            elif self.threshold_mask is not None:
                import numpy as np
                total_pixels = np.sum(self.threshold_mask > 0)
                self.total_pixels_var.set(str(total_pixels))
            else:
                self.total_pixels_var.set("0")

        except Exception as e:
            print(f"[ERROR] 统计计算失败: {e}")

    def update_overlay(self):
        """更新叠加图像"""
        try:
            if not hasattr(self, 'current_roi3_coords') or self.current_roi3_coords is None:
                return

            # 确定使用哪个mask
            if hasattr(self, 'current_mask_for_overlay') and self.current_mask_for_overlay is not None:
                mask = self.current_mask_for_overlay
            elif self.threshold_mask is not None:
                mask = self.threshold_mask
            else:
                return

            # 创建叠加
            overlay = self.create_overlay_image(
                self.roi1_image,
                mask,
                self.current_roi3_coords,
                self.overlay_alpha.get()
            )

            if overlay is not None:
                self.current_overlay_image = overlay
                self.update_roi_canvas_with_overlay()

        except Exception as e:
            print(f"[ERROR] 叠加更新失败: {e}")

    def update_heat_map_overlay(self):
        """更新热力图叠加图像"""
        try:
            if not hasattr(self, 'current_roi3_coords') or self.current_roi3_coords is None:
                return

            if self.heat_map is None:
                return

            # 创建热力图叠加
            overlay = self.create_heat_map_overlay_image(
                self.roi1_image,
                self.heat_map,
                self.current_roi3_coords,
                self.heatmap_alpha_var.get()
            )

            if overlay is not None:
                self.current_overlay_image = overlay
                self.update_roi_canvas_with_overlay()

        except Exception as e:
            print(f"[ERROR] 热力图叠加更新失败: {e}")

    def get_y_zoom_info(self):
        """获取当前Y轴缩放信息"""
        return {
            'zoom_factor': self.y_zoom_factor,
            'zoom_percentage': int(self.y_zoom_factor * 100),
            'is_zoomed': self.y_zoom_factor != 1.0,
            'can_zoom_in': self.y_zoom_factor < self.y_max_zoom,
            'can_zoom_out': self.y_zoom_factor > self.y_min_zoom
        }

    def extract_roi3_from_roi1(self, roi1_image, roi_config):
        """从ROI1图像中提取ROI3区域"""
        try:
            if roi1_image is None:
                return None, None

            # 获取ROI1尺寸
            roi1_width, roi1_height = roi1_image.size

            # 计算ROI3区域中心（使用图像中心作为交点）
            center_x = roi1_width // 2
            center_y = roi1_height // 2

            # 从配置中获取ROI3扩展参数
            roi3_params = roi_config['roi3']
            left = roi3_params[0]   # 左扩展
            right = roi3_params[1]  # 右扩展
            top = roi3_params[2]    # 上扩展
            bottom = roi3_params[3]  # 下扩展

            # 计算ROI3区域坐标
            roi3_left = center_x - left
            roi3_top = center_y - top
            roi3_right = center_x + right
            roi3_bottom = center_y + bottom

            # 确保坐标在图像边界内
            roi3_left = max(0, roi3_left)
            roi3_top = max(0, roi3_top)
            roi3_right = min(roi1_width, roi3_right)
            roi3_bottom = min(roi1_height, roi3_bottom)

            # 提取ROI3区域
            roi3_region = roi1_image.crop((roi3_left, roi3_top, roi3_right, roi3_bottom))

            return roi3_region, (roi3_left, roi3_top, roi3_right, roi3_bottom)

        except Exception as e:
            print(f"[ERROR] ROI3提取失败: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def apply_threshold_extraction(self, roi3_image, lower_thresh, upper_thresh):
        """对ROI3图像应用阈值范围提取，生成二进制mask"""
        try:
            import cv2
            import numpy as np

            if roi3_image is None:
                return None

            # 转换为灰度图（如果还不是）
            if roi3_image.mode != 'L':
                roi3_gray = roi3_image.convert('L')
            else:
                roi3_gray = roi3_image

            # 转换为numpy数组
            roi3_array = np.array(roi3_gray)

            # 应用阈值范围
            mask = cv2.inRange(roi3_array, lower_thresh, upper_thresh)

            return mask

        except Exception as e:
            print(f"[ERROR] 阈值提取失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def apply_heat_map_extraction(self, roi3_image):
        """对ROI3图像应用热力图提取，生成彩色热力图"""
        try:
            import numpy as np

            if roi3_image is None:
                return None

            # 转换为灰度图（如果还不是）
            if roi3_image.mode != 'L':
                roi3_gray = roi3_image.convert('L')
            else:
                roi3_gray = roi3_image

            # 转换为numpy数组
            roi3_array = np.array(roi3_gray)

            # 创建RGB彩色热力图
            heat_map = np.zeros((roi3_array.shape[0], roi3_array.shape[1], 3), dtype=np.uint8)

            # 定义颜色映射分段点：蓝(0-63)→黄(64-127)→橙(128-191)→红(192-255)
            blue_mask = roi3_array <= 63
            yellow_mask = (roi3_array >= 64) & (roi3_array <= 127)
            orange_mask = (roi3_array >= 128) & (roi3_array <= 191)
            red_mask = roi3_array >= 192

            # 应用颜色映射
            heat_map[blue_mask] = [0, 0, 255]      # 蓝色
            heat_map[yellow_mask] = [255, 255, 0]  # 黄色
            heat_map[orange_mask] = [255, 165, 0]  # 橙色
            heat_map[red_mask] = [255, 0, 0]      # 红色

            return heat_map

        except Exception as e:
            print(f"[ERROR] 热力图生成失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def analyze_connected_components(self, threshold_mask):
        """对阈值mask进行连通域分析，返回最大连通域mask和统计信息"""
        try:
            import cv2
            import numpy as np

            if threshold_mask is None:
                return None, 0, 0

            # 执行连通域分析
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                threshold_mask, connectivity=8, ltype=cv2.CV_32S
            )

            # 查找最大连通域（排除背景标签0）
            if num_labels > 1:
                largest_area = 0
                largest_label = 1

                for i in range(1, num_labels):
                    area = stats[i, cv2.CC_STAT_AREA]
                    if area > largest_area:
                        largest_area = area
                        largest_label = i

                # 创建仅包含最大连通域的mask
                largest_component_mask = np.zeros_like(threshold_mask)
                largest_component_mask[labels == largest_label] = 255

                total_pixels = np.sum(threshold_mask > 0)
                component_count = num_labels - 1

                return largest_component_mask, largest_area, component_count
            else:
                return None, 0, 0

        except Exception as e:
            print(f"[ERROR] 连通域分析失败: {e}")
            import traceback
            traceback.print_exc()
            return None, 0, 0

    def create_overlay_image(self, base_image, mask, roi3_coords, alpha=0.5):
        """创建半透明叠加图像用于画布显示"""
        try:
            import numpy as np
            from PIL import ImageDraw

            if mask is None or roi3_coords is None:
                return None

            # 创建与基础图像相同大小的叠加图像
            overlay = Image.new('RGBA', base_image.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)

            roi3_left, roi3_top, roi3_right, roi3_bottom = roi3_coords
            roi3_width = roi3_right - roi3_left
            roi3_height = roi3_bottom - roi3_top

            # 将mask调整为ROI3尺寸
            mask_pil = Image.fromarray(mask)
            mask_resized = mask_pil.resize((roi3_width, roi3_height), Image.Resampling.NEAREST)

            # 创建带透明度的彩色叠加
            overlay_color = (255, 255, 0, int(255 * alpha))  # 黄色带alpha

            # 在mask激活区域应用叠加
            mask_array = np.array(mask_resized)
            for y in range(roi3_height):
                for x in range(roi3_width):
                    if mask_array[y, x] > 0:
                        overlay_draw.point((roi3_left + x, roi3_top + y), fill=overlay_color)

            return overlay

        except Exception as e:
            print(f"[ERROR] 叠加图像创建失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def create_heat_map_overlay_image(self, base_image, heat_map, roi3_coords, alpha=0.6):
        """创建热力图叠加图像用于画布显示"""
        try:
            import numpy as np
            from PIL import ImageDraw

            if heat_map is None or roi3_coords is None:
                return None

            # 创建与基础图像相同大小的叠加图像
            overlay = Image.new('RGBA', base_image.size, (0, 0, 0, 0))

            roi3_left, roi3_top, roi3_right, roi3_bottom = roi3_coords
            roi3_width = roi3_right - roi3_left
            roi3_height = roi3_bottom - roi3_top

            # 将热力图调整为ROI3尺寸
            heat_map_pil = Image.fromarray(heat_map)
            heat_map_resized = heat_map_pil.resize((roi3_width, roi3_height), Image.Resampling.NEAREST)

            # 转换为RGBA并添加透明度
            heat_map_rgba = heat_map_resized.convert('RGBA')

            # 创建透明度蒙版
            alpha_array = np.full((roi3_height, roi3_width), int(255 * alpha), dtype=np.uint8)
            alpha_pil = Image.fromarray(alpha_array)

            # 将alpha通道应用到热力图
            r, g, b, a = heat_map_rgba.split()
            heat_map_with_alpha = Image.merge('RGBA', (r, g, b, alpha_pil))

            # 将热力图叠加到基础图像上
            overlay.paste(heat_map_with_alpha, (roi3_left, roi3_top), heat_map_with_alpha)

            return overlay

        except Exception as e:
            print(f"[ERROR] 热力图叠加图像创建失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def update_roi_canvas_with_overlay(self):
        """更新ROI1画布叠加显示（支持热力图和阈值mask的互斥显示）"""
        try:
            # 首先调用现有的可视化
            self.update_roi_visualization()

            # 检查是否显示叠加（热力图模式或阈值mask模式）
            show_overlay = False

            # 热力图模式：优先检查是否在热力图模式
            if hasattr(self, 'heatmap_mode') and self.heatmap_mode and self.heat_map is not None:
                show_overlay = True
            # 阈值mask模式：检查是否启用阈值叠加
            elif self.overlay_enabled.get() and hasattr(self, 'current_overlay_image') and self.current_overlay_image:
                show_overlay = True

            # 如果显示叠加且有叠加图像，则添加到画布
            if show_overlay:
                # 获取画布尺寸
                canvas_width = self.roi_canvas.winfo_width()
                canvas_height = self.roi_canvas.winfo_height()

                if canvas_width <= 1 or canvas_height <= 1:
                    canvas_width = 640
                    canvas_height = 900

                # 计算缩放因子（重用现有逻辑）
                roi_config = self.get_roi_config_values()
                roi1_x1, roi1_y1, roi1_x2, roi1_y2 = roi_config['roi1']
                roi1_width = roi1_x2 - roi1_x1
                roi1_height = roi1_y2 - roi1_y1

                scale_x = (canvas_width - 20) / roi1_width
                scale_y = (canvas_height - 20) / roi1_height
                scale = min(scale_x, scale_y, 1.0)

                x_offset = (canvas_width - roi1_width * scale) // 2
                y_offset = (canvas_height - roi1_height * scale) // 2

                # 转换叠加图像为PhotoImage
                overlay_photo = ImageTk.PhotoImage(self.current_overlay_image)

                # 在画布上绘制叠加
                self.roi_canvas.create_image(
                    canvas_width // 2, canvas_height // 2,
                    image=overlay_photo,
                    anchor=tk.CENTER,
                    tags="overlay"
                )

                # 存储引用以防止垃圾回收
                self.current_overlay_photo = overlay_photo

        except Exception as e:
            print(f"[ERROR] 画布叠加更新失败: {e}")
            import traceback
            traceback.print_exc()

    def detect_image_sequence(self, current_file):
        """检测当前图片所在的序列 - 改进的数字索引版本"""
        try:
            import glob

            # 获取当前文件的目录和文件名
            dir_path = os.path.dirname(current_file)
            filename = os.path.basename(current_file)

            print(f"[DEBUG] 分析文件名: {filename}")

            # 提取数字序列 - 优先提取最长的数字序列
            import re

            # 找到所有数字序列，选择最长的（避免匹配roi1中的'1'）
            all_matches = re.findall(r'(\d+)', filename)
            if not all_matches:
                print(f"[ERROR] 文件名中未找到数字: {filename}")
                self.current_image_sequence = []
                self.current_image_index = -1
                return

            # 选择最长的数字序列
            longest_match = max(all_matches, key=len)
            match = re.search(r'(' + longest_match + ')', filename)
            if not match:
                print(f"[ERROR] 无法重新匹配最长数字序列: {filename}")
                self.current_image_sequence = []
                self.current_image_index = -1
                return

            sequence_number = int(match.group(1))
            print(f"[DEBUG] 当前文件序列号: {sequence_number}")

            # 构建搜索模式：替换数字为通配符
            pattern_prefix = filename[:match.start()]
            pattern_suffix = filename[match.end():]
            search_pattern = f"{pattern_prefix}*{pattern_suffix}"

            print(f"[DEBUG] 搜索模式: {search_pattern}")

            # 搜索匹配的文件
            search_path = os.path.join(dir_path, search_pattern)
            found_files = glob.glob(search_path)

            print(f"[DEBUG] 找到匹配文件: {len(found_files)}")

            # 提取并排序文件
            sequence_files = []
            for file_path in found_files:
                file_name = os.path.basename(file_path)
                # 在相同位置查找相同长度的数字
                file_match = re.search(r'(\d{' + str(len(longest_match)) + r'})', file_name[match.start():])
                if file_match:
                    file_number = int(file_match.group())
                    sequence_files.append((file_number, file_path))

            # 按数字大小排序
            sequence_files.sort(key=lambda x: x[0])
            sorted_sequence = [path for _, path in sequence_files]

            # 找到当前文件的位置 - 处理路径差异
            try:
                # 标准化路径进行比较
                normalized_current = os.path.normpath(current_file)
                normalized_sequence = [os.path.normpath(path) for path in sorted_sequence]

                current_index = normalized_sequence.index(normalized_current)
                print(f"[SUCCESS] 序列长度: {len(sorted_sequence)}, 当前索引: {current_index}")

                self.current_image_sequence = sorted_sequence
                self.current_image_index = current_index

                print(f"[DEBUG] 检测到完整序列:")
                for i, file_path in enumerate(sorted_sequence):
                    print(f"[DEBUG] {i:2d}: {os.path.basename(file_path)}")
                print(f"[INFO] 当前图片: {os.path.basename(current_file)} (索引 {current_index})")

            except ValueError:
                print(f"[ERROR] 当前文件不在序列中: {filename}")
                print(f"[DEBUG] 当前文件: {os.path.normpath(current_file)}")
                print(f"[DEBUG] 序列文件: {[os.path.basename(p) for p in sorted_sequence[:5]]}...")
                self.current_image_sequence = []
                self.current_image_index = -1

        except Exception as e:
            print(f"[ERROR] 序列检测失败: {e}")
            import traceback
            traceback.print_exc()
            self.current_image_sequence = []
            self.current_image_index = -1

    def load_image_by_index(self, index):
        """根据索引加载图片"""
        if not self.current_image_sequence or index < 0 or index >= len(self.current_image_sequence):
            return False

        try:
            new_file = self.current_image_sequence[index]

            # 加载新图片
            new_image = Image.open(new_file)

            # 更新当前状态
            self.roi1_image = new_image
            self.current_roi1_path = new_file
            self.current_image_index = index

            # 更新UI显示
            self.image_path_var.set(f"已加载: {os.path.basename(new_file)}")
            self.update_roi_visualization()

            # 连续检查：自动执行阈值处理
            if self.continuous_check_enabled.get():
                self.root.after(200, self.auto_execute_threshold_processing)

            # 连续热力图：自动执行热力图处理（独立于连续检查）
            if self.continuous_heatmap_enabled.get():
                self.root.after(300, self.auto_execute_heatmap_processing)

            # 更新状态栏
            sequence_info = f" | 序列: {index+1}/{len(self.current_image_sequence)}"
            self.status_var.set(f"ROI1图片已加载: {os.path.basename(new_file)}{sequence_info} (按D/A或→/←切换)")

            print(f"[INFO] 加载图片: {os.path.basename(new_file)} ({index+1}/{len(self.current_image_sequence)})")
            return True

        except Exception as e:
            print(f"[ERROR] 加载图片失败: {e}")
            self.status_var.set(f"加载图片失败: {str(e)}")
            return False

    def on_key_press(self, event):
        """处理键盘按键事件"""
        try:
            key = event.keysym.lower()
            print(f"[DEBUG] 按键事件: {key}, 序列长度: {len(self.current_image_sequence)}, 当前索引: {self.current_image_index}")

            # 调试：显示所有按键（临时）
            if key not in ['d', 'right', 'a', 'left']:
                print(f"[DEBUG] 忽略按键: {key}")
                return

            # 只在有图片序列时处理导航键
            if not self.current_image_sequence or len(self.current_image_sequence) <= 1:
                print(f"[DEBUG] 无有效图片序列，忽略按键: {key}")
                self.status_var.set("没有检测到图片序列，无法使用键盘导航")
                return

            # 处理导航按键
            if key in ['d', 'right']:  # D 或 → : 下一张图片
                print(f"[DEBUG] 尝试加载下一张图片，当前索引: {self.current_image_index}")
                if self.current_image_index < len(self.current_image_sequence) - 1:
                    success = self.load_image_by_index(self.current_image_index + 1)
                    print(f"[DEBUG] 下一张图片加载结果: {success}")
                else:
                    self.status_var.set("已经是最后一张图片")
                    print(f"[DEBUG] 已经是最后一张图片")

            elif key in ['a', 'left']:  # A 或 ← : 上一张图片
                print(f"[DEBUG] 尝试加载上一张图片，当前索引: {self.current_image_index}")
                if self.current_image_index > 0:
                    success = self.load_image_by_index(self.current_image_index - 1)
                    print(f"[DEBUG] 上一张图片加载结果: {success}")
                else:
                    self.status_var.set("已经是第一张图片")
                    print(f"[DEBUG] 已经是第一张图片")

        except Exception as e:
            print(f"[ERROR] 处理键盘事件失败: {e}")
            import traceback
            traceback.print_exc()

def main():
    """主函数"""
    root = tk.Tk()

    # 设置窗口标题
    root.title("SimpleFEM 配置管理器")

    # 自动全屏
    root.state('zoomed')  # Windows全屏
    # 如果在其他平台上，可以使用以下方式：
    # root.attributes('-zoomed', True)  # 跨平台全屏

    print("[INFO] 程序已启动并自动全屏")
    print("[INFO] 按 F11 键可以切换全屏模式")

    app = SimpleFEMConfigGUI(root)

    # 设置窗口图标（如果有的话）
    # root.iconbitmap('icon.ico')

    # 添加F11键切换全屏功能
    def toggle_fullscreen(event=None):
        current_state = root.state()
        if current_state == 'zoomed':
            root.state('normal')  # 退出全屏
            print("[INFO] 已退出全屏模式")
        else:
            root.state('zoomed')  # 进入全屏
            print("[INFO] 已进入全屏模式")

    # 绑定F11键切换全屏
    root.bind('<F11>', toggle_fullscreen)

    # 添加ESC键退出全屏功能
    def exit_fullscreen(event=None):
        if root.state() == 'zoomed':
            root.state('normal')
            print("[INFO] 按ESC键退出全屏模式")

    root.bind('<Escape>', exit_fullscreen)

    # 运行主循环
    root.mainloop()

if __name__ == "__main__":
    main()