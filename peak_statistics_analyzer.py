"""
Peak Statistics CSV Analyzer - GUI程序

用于分析SimpleFEM生成的peak_statistics CSV文件
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import os

class PeakStatisticsAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("Peak Statistics Analyzer")
        self.root.geometry("1400x900")

        self.df = None
        self.csv_path = None

        # 定义所有可用的列
        self.all_columns = {
            "frame_index": {"name": "帧索引", "width": 80, "visible": True},
            "peak_type": {"name": "波峰类型", "width": 80, "visible": True},
            "frame_diff": {"name": "帧差值", "width": 80, "visible": False},
            "pre_peak_avg": {"name": "前置均值", "width": 80, "visible": False},
            "post_peak_avg": {"name": "后置均值", "width": 80, "visible": False},
            "difference_threshold_used": {"name": "差值阈值", "width": 90, "visible": False},
            "threshold_used": {"name": "使用阈值", "width": 80, "visible": False},
            "bg_mean": {"name": "背景均值", "width": 80, "visible": False},
            "peak_max_value": {"name": "峰值", "width": 80, "visible": False},
            "roi3_peak_max_value": {"name": "ROI3峰值", "width": 90, "visible": False},
            "roi3_peak_max_frame": {"name": "ROI3峰值帧", "width": 100, "visible": False},
            "pre_peak_frame_start": {"name": "前置帧起", "width": 80, "visible": False},
            "pre_peak_frame_end": {"name": "前置帧止", "width": 80, "visible": False},
            "post_peak_frame_start": {"name": "后置帧起", "width": 80, "visible": False},
            "post_peak_frame_end": {"name": "后置帧止", "width": 80, "visible": False},
            "g1_value": {"name": "G1值", "width": 80, "visible": True},
            "g2_value": {"name": "G2值", "width": 80, "visible": True},
            "g1_g2_override_applied": {"name": "G1/G2覆盖", "width": 90, "visible": True},
            "g1_g2_override_frame_idx": {"name": "覆盖帧索引", "width": 100, "visible": True},
            "column_diff_value": {"name": "列灰度差值", "width": 100, "visible": True},
            "column_diff_override_applied": {"name": "列差覆盖", "width": 80, "visible": True},
            "column_diff_override_frame_idx": {"name": "列差覆盖帧", "width": 110, "visible": True},
        }

        self.create_widgets()

    def create_widgets(self):
        """创建GUI组件"""
        # 顶部工具栏
        toolbar = ttk.Frame(self.root, padding=5)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="打开CSV文件", command=self.load_csv).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="刷新统计", command=self.refresh_stats).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="生成图表", command=self.show_charts).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="导出报告", command=self.export_report).pack(side=tk.LEFT, padx=5)

        # 统计信息面板
        stats_frame = ttk.LabelFrame(self.root, text="统计信息", padding=5)
        stats_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        self.stats_labels = {}
        stats_vars = [
            "总波峰数", "绿色波峰", "红色波峰",
            "G1/G2覆盖", "列灰度差值覆盖",
            "平均G1值", "平均G2值", "平均列灰度差值"
        ]

        for i, var in enumerate(stats_vars):
            lbl = ttk.Label(stats_frame, text=f"{var}: -", font=('Arial', 9))
            lbl.grid(row=i//4, column=i%4, padx=10, pady=2, sticky=tk.W)
            self.stats_labels[var] = lbl

        # 筛选面板
        filter_frame = ttk.LabelFrame(self.root, text="数据筛选", padding=5)
        filter_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # 第一行筛选
        row1 = ttk.Frame(filter_frame)
        row1.pack(fill=tk.X, pady=2)

        ttk.Label(row1, text="波峰类型:").pack(side=tk.LEFT)
        self.peak_type_var = tk.StringVar(value="全部")
        peak_type_cb = ttk.Combobox(row1, textvariable=self.peak_type_var,
                                     values=["全部", "green", "red"], width=10)
        peak_type_cb.pack(side=tk.LEFT, padx=5)
        peak_type_cb.bind("<<ComboboxSelected>>", self.apply_filters)

        ttk.Label(row1, text="G1/G2覆盖:").pack(side=tk.LEFT, padx=(20, 0))
        self.override_var = tk.StringVar(value="全部")
        override_cb = ttk.Combobox(row1, textvariable=self.override_var,
                                   values=["全部", "True", "False"], width=10)
        override_cb.pack(side=tk.LEFT, padx=5)
        override_cb.bind("<<ComboboxSelected>>", self.apply_filters)

        ttk.Button(row1, text="应用筛选", command=self.apply_filters).pack(side=tk.LEFT, padx=20)
        ttk.Button(row1, text="重置筛选", command=self.reset_filters).pack(side=tk.LEFT)
        ttk.Button(row1, text="选择列", command=self.show_column_selector).pack(side=tk.LEFT, padx=20)

        # 数据表格
        table_frame = ttk.LabelFrame(self.root, text="数据详情", padding=5)
        table_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 创建Treeview - 使用可配置的列
        visible_columns = [col for col, config in self.all_columns.items() if config["visible"]]
        self.tree = ttk.Treeview(table_frame, columns=visible_columns, show='headings', height=15)

        # 配置可见列
        for col in visible_columns:
            config = self.all_columns[col]
            self.tree.heading(col, text=config["name"])
            self.tree.column(col, width=config["width"])

        # 添加滚动条
        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(table_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)

        # 底部状态栏
        self.status_var = tk.StringVar(value="请打开CSV文件")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def load_csv(self):
        """加载CSV文件"""
        file_path = filedialog.askopenfilename(
            title="选择Peak Statistics CSV文件",
            filetypes=[("CSV文件", "*.csv"), ("所有文件", "*.*")]
        )

        if not file_path:
            return

        try:
            self.df = pd.read_csv(file_path)
            self.csv_path = file_path
            self.status_var.set(f"已加载: {os.path.basename(file_path)} ({len(self.df)} 行)")

            self.reset_filters()
            self.refresh_stats()

        except Exception as e:
            messagebox.showerror("错误", f"无法读取文件:\n{str(e)}")

    def reset_filters(self):
        """重置筛选"""
        if self.df is None:
            return

        self.filtered_df = self.df.copy()
        self.update_table()
        self.status_var.set(f"显示 {len(self.filtered_df)} / {len(self.df)} 行")

    def apply_filters(self, event=None):
        """应用筛选条件"""
        if self.df is None:
            return

        df = self.df.copy()

        # 波峰类型筛选
        peak_type = self.peak_type_var.get()
        if peak_type != "全部":
            df = df[df['peak_type'] == peak_type]

        # G1/G2覆盖筛选
        override = self.override_var.get()
        if override != "全部":
            if override == "True":
                df = df[df['g1_g2_override_applied'] == True]
            else:
                df = df[df['g1_g2_override_applied'] == False]

        self.filtered_df = df
        self.update_table()
        self.status_var.set(f"筛选结果: {len(self.filtered_df)} / {len(self.df)} 行")

    def update_table(self):
        """更新数据表格"""
        if self.filtered_df is None:
            return

        # 清空表格
        for item in self.tree.get_children():
            self.tree.delete(item)

        # 获取当前可见的列
        visible_columns = [col for col, config in self.all_columns.items() if config["visible"]]

        # 插入数据
        for idx, row in self.filtered_df.iterrows():
            values = []
            for col in visible_columns:
                val = row.get(col, "")
                if pd.isna(val):
                    val = ""
                elif isinstance(val, float):
                    val = f"{val:.2f}" if val != 0 else ""
                elif isinstance(val, int):
                    val = str(val)
                values.append(str(val))

            self.tree.insert("", tk.END, values=values)

    def show_column_selector(self):
        """显示列选择对话框"""
        selector = tk.Toplevel(self.root)
        selector.title("选择显示的列")
        selector.geometry("400x600")
        selector.transient(self.root)
        selector.grab_set()

        # 标题
        ttk.Label(selector, text="勾选要显示的列:", font=('Arial', 10, 'bold')).pack(pady=10)

        # 创建滚动框架
        frame = ttk.Frame(selector)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        canvas = tk.Canvas(frame)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # 复选框变量字典
        self.column_vars = {}

        # 为每一列创建复选框
        for col_id, config in self.all_columns.items():
            var = tk.BooleanVar(value=config["visible"])
            self.column_vars[col_id] = var

            frame_cb = ttk.Frame(scrollable_frame)
            frame_cb.pack(fill=tk.X, pady=2)

            cb = ttk.Checkbutton(frame_cb, variable=var, text=config["name"])
            cb.pack(side=tk.LEFT, padx=5)

            # 显示列宽度信息
            ttk.Label(frame_cb, text=f"({config['width']}px)", font=('Arial', 8)).pack(side=tk.LEFT)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 按钮区域
        button_frame = ttk.Frame(selector)
        button_frame.pack(fill=tk.X, padx=10, pady=10)

        def apply_column_selection():
            """应用列选择"""
            for col_id, var in self.column_vars.items():
                self.all_columns[col_id]["visible"] = var.get()

            # 重建Treeview
            self.rebuild_treeview()
            selector.destroy()

        def select_all():
            """全选"""
            for var in self.column_vars.values():
                var.set(True)

        def deselect_all():
            """全不选"""
            for var in self.column_vars.values():
                var.set(False)

        def select_basic():
            """选择基础列"""
            for col_id in ["frame_index", "peak_type", "g1_value", "g2_value",
                          "g1_g2_override_applied", "column_diff_value"]:
                if col_id in self.column_vars:
                    self.column_vars[col_id].set(True)
            for col_id in self.column_vars:
                if col_id not in ["frame_index", "peak_type", "g1_value", "g2_value",
                                   "g1_g2_override_applied", "column_diff_value"]:
                    self.column_vars[col_id].set(False)

        ttk.Button(button_frame, text="全选", command=select_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="全不选", command=deselect_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="基础列", command=select_basic).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="应用", command=apply_column_selection).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="取消", command=selector.destroy).pack(side=tk.RIGHT)

    def rebuild_treeview(self):
        """重建Treeview以应用列配置"""
        # 获取当前表格的父容器
        table_frame = self.tree.master

        # 销毁旧的Treeview和滚动条
        for widget in table_frame.winfo_children():
            widget.destroy()

        # 创建新的Treeview
        visible_columns = [col for col, config in self.all_columns.items() if config["visible"]]
        self.tree = ttk.Treeview(table_frame, columns=visible_columns, show='headings', height=15)

        # 配置可见列
        for col in visible_columns:
            config = self.all_columns[col]
            self.tree.heading(col, text=config["name"])
            self.tree.column(col, width=config["width"])

        # 重新添加滚动条
        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(table_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)

        # 刷新数据显示
        if self.df is not None:
            self.update_table()

    def refresh_stats(self):
        """刷新统计信息"""
        if self.df is None:
            return

        df = self.df

        # 基础统计
        total = len(df)
        green_count = len(df[df['peak_type'] == 'green'])
        red_count = len(df[df['peak_type'] == 'red'])

        # 覆盖统计
        g1_g2_override_count = len(df[df['g1_g2_override_applied'] == True])
        column_diff_override_count = len(df[df['column_diff_override_applied'] == True])

        # 平均值统计（忽略空值）
        avg_g1 = df['g1_value'].dropna().mean() if 'g1_value' in df.columns else 0
        avg_g2 = df['g2_value'].dropna().mean() if 'g2_value' in df.columns else 0
        avg_column_diff = df['column_diff_value'].dropna().mean() if 'column_diff_value' in df.columns else 0

        # 更新标签
        self.stats_labels["总波峰数"].config(text=f"总波峰数: {total}")
        self.stats_labels["绿色波峰"].config(text=f"绿色波峰: {green_count} ({green_count/total*100:.1f}%)")
        self.stats_labels["红色波峰"].config(text=f"红色波峰: {red_count} ({red_count/total*100:.1f}%)")
        self.stats_labels["G1/G2覆盖"].config(text=f"G1/G2覆盖: {g1_g2_override_count} ({g1_g2_override_count/total*100:.1f}%)")
        self.stats_labels["列灰度差值覆盖"].config(text=f"列灰度差值覆盖: {column_diff_override_count} ({column_diff_override_count/total*100:.1f}%)")
        self.stats_labels["平均G1值"].config(text=f"平均G1值: {avg_g1:.2f}%")
        self.stats_labels["平均G2值"].config(text=f"平均G2值: {avg_g2:.2f}%")
        self.stats_labels["平均列灰度差值"].config(text=f"平均列灰度差值: {avg_column_diff:.2f}")

    def show_charts(self):
        """显示图表"""
        if self.df is None:
            messagebox.showwarning("警告", "请先加载CSV文件")
            return

        chart_window = tk.Toplevel(self.root)
        chart_window.title("数据图表")
        chart_window.geometry("1000x700")

        # 创建图表
        fig = Figure(figsize=(10, 8))

        # 1. 波峰类型分布饼图
        ax1 = fig.add_subplot(221)
        green_count = len(self.df[self.df['peak_type'] == 'green'])
        red_count = len(self.df[self.df['peak_type'] == 'red'])
        ax1.pie([green_count, red_count], labels=['绿色', '红色'], autopct='%1.1f%%', colors=['green', 'red'])
        ax1.set_title('波峰类型分布')

        # 2. G1/G2值散点图
        ax2 = fig.add_subplot(222)
        df_valid = self.df.dropna(subset=['g1_value', 'g2_value'])
        ax2.scatter(df_valid['g1_value'], df_valid['g2_value'],
                   c=df_valid['peak_type'].map({'green': 'green', 'red': 'red'}),
                   alpha=0.6)
        ax2.set_xlabel('G1值 (%)')
        ax2.set_ylabel('G2值 (%)')
        ax2.set_title('G1/G2分布')
        ax2.grid(True, alpha=0.3)

        # 3. 覆盖情况柱状图
        ax3 = fig.add_subplot(223)
        override_counts = [
            len(self.df[self.df['g1_g2_override_applied'] == True]),
            len(self.df[self.df['column_diff_override_applied'] == True])
        ]
        ax3.bar(['G1/G2覆盖', '列灰度差值覆盖'], override_counts, color=['blue', 'orange'])
        ax3.set_ylabel('波峰数')
        ax3.set_title('覆盖逻辑应用次数')
        for i, v in enumerate(override_counts):
            ax3.text(i, v, str(v), ha='center', va='bottom')

        # 4. 列灰度差值分布直方图
        ax4 = fig.add_subplot(224)
        if 'column_diff_value' in self.df.columns:
            diff_data = self.df['column_diff_value'].dropna()
            ax4.hist(diff_data, bins=20, color='orange', alpha=0.7, edgecolor='black')
            ax4.set_xlabel('列灰度差值')
            ax4.set_ylabel('频数')
            ax4.set_title('列灰度差值分布')
            ax4.axvline(x=15.0, color='red', linestyle='--', label='阈值(15.0)')
            ax4.legend()

        plt.tight_layout()

        # 嵌入到Tkinter
        canvas = FigureCanvasTkAgg(fig, master=chart_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def export_report(self):
        """导出分析报告"""
        if self.df is None:
            messagebox.showwarning("警告", "请先加载CSV文件")
            return

        file_path = filedialog.asksaveasfilename(
            title="保存分析报告",
            defaultextension=".txt",
            filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")]
        )

        if not file_path:
            return

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write("Peak Statistics Analysis Report\n")
                f.write("=" * 60 + "\n\n")

                f.write(f"文件: {self.csv_path}\n")
                f.write(f"总波峰数: {len(self.df)}\n\n")

                # 波峰类型统计
                f.write("-" * 40 + "\n")
                f.write("波峰类型统计\n")
                f.write("-" * 40 + "\n")
                green_count = len(self.df[self.df['peak_type'] == 'green'])
                red_count = len(self.df[self.df['peak_type'] == 'red'])
                f.write(f"绿色波峰: {green_count} ({green_count/len(self.df)*100:.1f}%)\n")
                f.write(f"红色波峰: {red_count} ({red_count/len(self.df)*100:.1f}%)\n\n")

                # 覆盖逻辑统计
                f.write("-" * 40 + "\n")
                f.write("覆盖逻辑统计\n")
                f.write("-" * 40 + "\n")
                g1_g2_override = len(self.df[self.df['g1_g2_override_applied'] == True])
                column_diff_override = len(self.df[self.df['column_diff_override_applied'] == True])
                f.write(f"G1/G2覆盖: {g1_g2_override} 次\n")
                f.write(f"列灰度差值覆盖: {column_diff_override} 次\n\n")

                # 数值统计
                f.write("-" * 40 + "\n")
                f.write("数值统计\n")
                f.write("-" * 40 + "\n")

                if 'g1_value' in self.df.columns:
                    g1_stats = self.df['g1_value'].describe()
                    f.write(f"G1值:\n")
                    f.write(f"  均值: {g1_stats['mean']:.2f}%\n")
                    f.write(f"  最小值: {g1_stats['min']:.2f}%\n")
                    f.write(f"  最大值: {g1_stats['max']:.2f}%\n\n")

                if 'g2_value' in self.df.columns:
                    g2_stats = self.df['g2_value'].describe()
                    f.write(f"G2值:\n")
                    f.write(f"  均值: {g2_stats['mean']:.2f}%\n")
                    f.write(f"  最小值: {g2_stats['min']:.2f}%\n")
                    f.write(f"  最大值: {g2_stats['max']:.2f}%\n\n")

                if 'column_diff_value' in self.df.columns:
                    diff_stats = self.df['column_diff_value'].describe()
                    f.write(f"列灰度差值:\n")
                    f.write(f"  均值: {diff_stats['mean']:.2f}\n")
                    f.write(f"  最小值: {diff_stats['min']:.2f}\n")
                    f.write(f"  最大值: {diff_stats['max']:.2f}\n\n")

            messagebox.showinfo("成功", f"报告已保存到:\n{file_path}")

        except Exception as e:
            messagebox.showerror("错误", f"无法保存报告:\n{str(e)}")


def main():
    root = tk.Tk()
    app = PeakStatisticsAnalyzer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
