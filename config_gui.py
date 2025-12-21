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

class SimpleFEMConfigGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SimpleFEM 配置管理器")
        self.root.geometry("800x700")

        # 配置文件路径
        self.config_path = "simple_fem_config.json"
        self.config_data = {}

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

        self.roi_vars = {}

        # ROI1配置
        ttk.Label(frame, text="ROI1 大区域配置", font=('Arial', 12, 'bold')).grid(row=0, column=0, columnspan=2, pady=(10, 5))

        roi1_configs = [
            ("roi_capture.default_config.x1", "X1坐标", "int"),
            ("roi_capture.default_config.y1", "Y1坐标", "int"),
            ("roi_capture.default_config.x2", "X2坐标", "int"),
            ("roi_capture.default_config.y2", "Y2坐标", "int"),
        ]

        row = 1
        for key, label, config_type in roi1_configs:
            ttk.Label(frame, text=f"  {label}").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
            var = tk.StringVar()
            ttk.Entry(frame, textvariable=var, width=15).grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
            self.roi_vars[key] = var
            row += 1

        # ROI2配置
        ttk.Label(frame, text="ROI2 小区域配置", font=('Arial', 12, 'bold')).grid(row=row, column=0, columnspan=2, pady=(10, 5))
        row += 1

        roi2_configs = [
            ("roi_capture.roi2_config.extension_params.left", "左边距", "int"),
            ("roi_capture.roi2_config.extension_params.right", "右边距", "int"),
            ("roi_capture.roi2_config.extension_params.top", "上边距", "int"),
            ("roi_capture.roi2_config.extension_params.bottom", "下边距", "int"),
        ]

        for key, label, config_type in roi2_configs:
            ttk.Label(frame, text=f"  {label}").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
            var = tk.StringVar()
            ttk.Entry(frame, textvariable=var, width=15).grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
            self.roi_vars[key] = var
            row += 1

        # ROI3配置
        ttk.Label(frame, text="ROI3 扩展区域配置", font=('Arial', 12, 'bold')).grid(row=row, column=0, columnspan=2, pady=(10, 5))
        row += 1

        roi3_configs = [
            ("roi_capture.roi3_config.extension_params.left", "左边距", "int"),
            ("roi_capture.roi3_config.extension_params.right", "右边距", "int"),
            ("roi_capture.roi3_config.extension_params.top", "上边距", "int"),
            ("roi_capture.roi3_config.extension_params.bottom", "下边距", "int"),
        ]

        for key, label, config_type in roi3_configs:
            ttk.Label(frame, text=f"  {label}").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
            var = tk.StringVar()
            ttk.Entry(frame, textvariable=var, width=15).grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
            self.roi_vars[key] = var
            row += 1

        # 其他ROI配置
        ttk.Label(frame, text="其他配置", font=('Arial', 12, 'bold')).grid(row=row, column=0, columnspan=2, pady=(10, 5))
        row += 1

        other_configs = [
            ("roi_capture.frame_rate", "采集帧率", "int"),
            ("roi3_config.save_roi3", "保存ROI3图像", "bool"),
        ]

        for key, label, config_type in other_configs:
            ttk.Label(frame, text=f"  {label}").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)

            if config_type == "bool":
                var = tk.BooleanVar()
                ttk.Checkbutton(frame, variable=var).grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
            else:
                var = tk.StringVar()
                ttk.Entry(frame, textvariable=var, width=15).grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)

            self.roi_vars[key] = var
            row += 1

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

def main():
    """主函数"""
    root = tk.Tk()
    app = SimpleFEMConfigGUI(root)

    # 设置窗口图标（如果有的话）
    # root.iconbitmap('icon.ico')

    # 运行主循环
    root.mainloop()

if __name__ == "__main__":
    main()