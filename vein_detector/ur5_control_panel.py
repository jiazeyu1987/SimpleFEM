#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UR5 机械臂控制面板 UI

提供图形化界面用于控制 UR5 机械臂。
UI 只负责显示和用户交互，所有控制逻辑通过 UR5Controller 类实现。
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import logging
from typing import Optional

from ur5_controller import (
    UR5Controller,
    Pose,
    JointAngles,
    ConnectionStatus,
    MoveType
)

from hotkey_manager import HotKeyManager, TopmostIndicator


logger = logging.getLogger(__name__)


class UR5ControlPanel:
    """
    UR5 机械臂控制面板 UI 类

    功能：
    - 连接/断开 UR5 机械臂
    - MOVEL/MOVEJ 移动控制
    - 参数调节（速度、加速度）
    - 实时状态显示
    - 数字/模拟输出控制
    """

    def __init__(self, root: tk.Tk):
        """
        初始化控制面板

        Args:
            root: Tkinter 根窗口
        """
        self.root = root
        self.root.title("UR5 机械臂控制面板")
        self.root.geometry("1000x700")

        # 创建控制器
        self.controller = UR5Controller()

        # 注册回调
        self.controller.add_status_callback(self._on_status_changed)
        self.controller.add_error_callback(self._on_error)

        # 创建快捷键管理器
        self.hotkey_manager = HotKeyManager(root)
        self.topmost_indicator = TopmostIndicator(root, "UR5 机械臂控制面板")

        # UI 变量
        self._create_ui_variables()

        # 创建界面
        self._create_widgets()

        # 注册快捷键
        self._setup_hotkeys()

        logger.info("UR5 控制面板初始化完成")

    def _create_ui_variables(self):
        """创建 UI 变量"""
        # 连接参数
        self.host_var = tk.StringVar(value=self.controller.DEFAULT_HOST)
        self.port_var = tk.IntVar(value=self.controller.DEFAULT_PORT)

        # 位姿参数
        self.pose_x = tk.DoubleVar(value=0.3)
        self.pose_y = tk.DoubleVar(value=0.2)
        self.pose_z = tk.DoubleVar(value=0.4)
        self.pose_rx = tk.DoubleVar(value=0.0)
        self.pose_ry = tk.DoubleVar(value=3.14)
        self.pose_rz = tk.DoubleVar(value=0.0)

        # 关节角度参数
        self.joint_j0 = tk.DoubleVar(value=0.0)
        self.joint_j1 = tk.DoubleVar(value=-1.57)
        self.joint_j2 = tk.DoubleVar(value=1.57)
        self.joint_j3 = tk.DoubleVar(value=-1.57)
        self.joint_j4 = tk.DoubleVar(value=-1.57)
        self.joint_j5 = tk.DoubleVar(value=0.0)

        # 运动参数
        self.acceleration = tk.DoubleVar(value=0.5)
        self.velocity = tk.DoubleVar(value=0.3)
        self.move_time = tk.DoubleVar(value=0.0)
        self.use_time = tk.BooleanVar(value=False)

        # 移动类型
        self.move_type = tk.StringVar(value="MOVEL")

        # 数字输出
        self.digital_outs = [tk.BooleanVar(value=False) for _ in range(8)]

        # 模拟输出
        self.analog_outs = [tk.DoubleVar(value=0.0) for _ in range(4)]

        # 状态显示
        self.status_text = tk.StringVar(value="未连接")
        self.current_pose_text = tk.StringVar(value="--")
        self.current_joints_text = tk.StringVar(value="--")

        # 置顶状态显示
        self.topmost_status_text = tk.StringVar(value="普通模式")

    def _create_widgets(self):
        """创建界面组件"""
        # 创建主框架
        main_frame = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 左侧面板
        left_frame = ttk.Frame(main_frame)
        main_frame.add(left_frame, weight=2)

        # 右侧面板
        right_frame = ttk.Frame(main_frame)
        main_frame.add(right_frame, weight=1)

        # 创建各个区域
        self._create_connection_area(left_frame)
        self._create_pose_control_area(left_frame)
        self._create_joint_control_area(left_frame)
        self._create_motion_params_area(left_frame)
        self._create_control_buttons(left_frame)

        # 右侧状态和IO
        self._create_status_area(right_frame)
        self._create_digital_io_area(right_frame)
        self._create_analog_io_area(right_frame)
        self._create_log_area(right_frame)

    def _create_connection_area(self, parent):
        """创建连接区域"""
        frame = ttk.LabelFrame(parent, text="连接设置", padding=10)
        frame.pack(fill=tk.X, padx=5, pady=5)

        # IP 地址
        ttk.Label(frame, text="IP 地址:").grid(row=0, column=0, sticky=tk.W, padx=5)
        ttk.Entry(frame, textvariable=self.host_var, width=20).grid(row=0, column=1, padx=5)

        # 端口
        ttk.Label(frame, text="端口:").grid(row=0, column=2, sticky=tk.W, padx=5)
        ttk.Entry(frame, textvariable=self.port_var, width=10).grid(row=0, column=3, padx=5)

        # 连接按钮
        self.connect_btn = ttk.Button(frame, text="连接", command=self._on_connect)
        self.connect_btn.grid(row=0, column=4, padx=5)

        self.disconnect_btn = ttk.Button(frame, text="断开", command=self._on_disconnect, state=tk.DISABLED)
        self.disconnect_btn.grid(row=0, column=5, padx=5)

    def _create_pose_control_area(self, parent):
        """创建位姿控制区域"""
        frame = ttk.LabelFrame(parent, text="笛卡尔位姿 (MOVEL)", padding=10)
        frame.pack(fill=tk.X, padx=5, pady=5)

        # 位姿输入框
        pose_vars = [
            ("X (m):", self.pose_x),
            ("Y (m):", self.pose_y),
            ("Z (m):", self.pose_z),
            ("RX (rad):", self.pose_rx),
            ("RY (rad):", self.pose_ry),
            ("RZ (rad):", self.pose_rz),
        ]

        for i, (label, var) in enumerate(pose_vars):
            row = i // 3
            col = (i % 3) * 2
            ttk.Label(frame, text=label).grid(row=row, column=col, sticky=tk.W, padx=5, pady=2)
            ttk.Entry(frame, textvariable=var, width=12).grid(row=row, column=col+1, padx=5, pady=2)

    def _create_joint_control_area(self, parent):
        """创建关节控制区域"""
        frame = ttk.LabelFrame(parent, text="关节角度 (MOVEJ)", padding=10)
        frame.pack(fill=tk.X, padx=5, pady=5)

        # 关节输入框
        joint_vars = [
            ("J0 (rad):", self.joint_j0),
            ("J1 (rad):", self.joint_j1),
            ("J2 (rad):", self.joint_j2),
            ("J3 (rad):", self.joint_j3),
            ("J4 (rad):", self.joint_j4),
            ("J5 (rad):", self.joint_j5),
        ]

        for i, (label, var) in enumerate(joint_vars):
            row = i // 3
            col = (i % 3) * 2
            ttk.Label(frame, text=label).grid(row=row, column=col, sticky=tk.W, padx=5, pady=2)
            ttk.Entry(frame, textvariable=var, width=12).grid(row=row, column=col+1, padx=5, pady=2)

    def _create_motion_params_area(self, parent):
        """创建运动参数区域"""
        frame = ttk.LabelFrame(parent, text="运动参数", padding=10)
        frame.pack(fill=tk.X, padx=5, pady=5)

        # 移动类型
        ttk.Label(frame, text="移动类型:").grid(row=0, column=0, sticky=tk.W, padx=5)
        type_frame = ttk.Frame(frame)
        type_frame.grid(row=0, column=1, columnspan=3, sticky=tk.W)
        ttk.Radiobutton(type_frame, text="MOVEL (直线)", variable=self.move_type, value="MOVEL").pack(side=tk.LEFT)
        ttk.Radiobutton(type_frame, text="MOVEJ (关节)", variable=self.move_type, value="MOVEJ").pack(side=tk.LEFT, padx=10)

        # 加速度
        ttk.Label(frame, text="加速度 (0-2):").grid(row=1, column=0, sticky=tk.W, padx=5)
        ttk.Scale(frame, from_=0.0, to=2.0, variable=self.acceleration, orient=tk.HORIZONTAL, length=150).grid(row=1, column=1, padx=5)
        ttk.Entry(frame, textvariable=self.acceleration, width=8).grid(row=1, column=2, padx=5)

        # 速度
        ttk.Label(frame, text="速度 (0-1):").grid(row=2, column=0, sticky=tk.W, padx=5)
        ttk.Scale(frame, from_=0.0, to=1.0, variable=self.velocity, orient=tk.HORIZONTAL, length=150).grid(row=2, column=1, padx=5)
        ttk.Entry(frame, textvariable=self.velocity, width=8).grid(row=2, column=2, padx=5)

        # 时间
        ttk.Checkbutton(frame, text="指定时间 (秒)", variable=self.use_time).grid(row=3, column=0, sticky=tk.W, padx=5)
        ttk.Entry(frame, textvariable=self.move_time, width=8).grid(row=3, column=1, sticky=tk.W, padx=5)

    def _create_control_buttons(self, parent):
        """创建控制按钮区域"""
        frame = ttk.LabelFrame(parent, text="控制", padding=10)
        frame.pack(fill=tk.X, padx=5, pady=5)

        # 移动按钮
        ttk.Button(frame, text="执行移动", command=self._on_move, width=15).grid(row=0, column=0, padx=5, pady=5)

        # 停止按钮
        ttk.Button(frame, text="停止 (关节)", command=self._on_stopj, width=15).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(frame, text="停止 (笛卡尔)", command=self._on_stopl, width=15).grid(row=0, column=2, padx=5, pady=5)

        # 快速预设
        preset_frame = ttk.Frame(frame)
        preset_frame.grid(row=1, column=0, columnspan=3, pady=5)

        ttk.Label(preset_frame, text="快速预设:").pack(side=tk.LEFT)
        ttk.Button(preset_frame, text="Home", command=self._preset_home, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(preset_frame, text="Zero", command=self._preset_zero, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(preset_frame, text="上方", command=self._preset_above, width=10).pack(side=tk.LEFT, padx=2)

    def _create_status_area(self, parent):
        """创建状态显示区域"""
        frame = ttk.LabelFrame(parent, text="状态", padding=10)
        frame.pack(fill=tk.X, padx=5, pady=5)

        # 连接状态
        status_frame = ttk.Frame(frame)
        status_frame.pack(fill=tk.X, pady=2)
        ttk.Label(status_frame, text="连接状态:").pack(side=tk.LEFT)
        ttk.Label(status_frame, textvariable=self.status_text, foreground="blue").pack(side=tk.LEFT, padx=5)

        # 当前位姿
        pose_frame = ttk.Frame(frame)
        pose_frame.pack(fill=tk.X, pady=2)
        ttk.Label(pose_frame, text="当前位姿:").pack(side=tk.LEFT)
        ttk.Label(pose_frame, textvariable=self.current_pose_text, foreground="green").pack(side=tk.LEFT, padx=5)

        # 当前关节
        joint_frame = ttk.Frame(frame)
        joint_frame.pack(fill=tk.X, pady=2)
        ttk.Label(joint_frame, text="当前关节:").pack(side=tk.LEFT)
        ttk.Label(joint_frame, textvariable=self.current_joints_text, foreground="green").pack(side=tk.LEFT, padx=5)

        # 窗口模式（置顶状态）
        topmost_frame = ttk.Frame(frame)
        topmost_frame.pack(fill=tk.X, pady=2)
        ttk.Label(topmost_frame, text="窗口模式:").pack(side=tk.LEFT)
        self.topmost_label = ttk.Label(topmost_frame, textvariable=self.topmost_status_text, foreground="purple")
        self.topmost_label.pack(side=tk.LEFT, padx=5)

        # 快捷键提示
        hint_frame = ttk.Frame(frame)
        hint_frame.pack(fill=tk.X, pady=2)
        ttk.Label(hint_frame, text="快捷键:", font=("Arial", 8)).pack(side=tk.LEFT)
        ttk.Label(hint_frame, text="Ctrl+T", font=("Arial", 8, "bold"), foreground="blue").pack(side=tk.LEFT, padx=2)
        ttk.Label(hint_frame, text="切换置顶", font=("Arial", 8)).pack(side=tk.LEFT)

    def _create_digital_io_area(self, parent):
        """创建数字IO区域"""
        frame = ttk.LabelFrame(parent, text="数字输出 (0-7)", padding=10)
        frame.pack(fill=tk.X, padx=5, pady=5)

        for i in range(8):
            var = self.digital_outs[i]
            ttk.Checkbutton(
                frame,
                text=f"D{i}",
                variable=var,
                command=lambda idx=i: self._on_digital_out_change(idx)
            ).grid(row=i//4, column=i%4, padx=5, pady=2)

    def _create_analog_io_area(self, parent):
        """创建模拟IO区域"""
        frame = ttk.LabelFrame(parent, text="模拟输出 (0-3)", padding=10)
        frame.pack(fill=tk.X, padx=5, pady=5)

        for i in range(4):
            row_frame = ttk.Frame(frame)
            row_frame.pack(fill=tk.X, pady=2)

            ttk.Label(row_frame, text=f"A{i}:").pack(side=tk.LEFT)

            var = self.analog_outs[i]
            scale = ttk.Scale(
                row_frame,
                from_=0.0,
                to=1.0,
                variable=var,
                orient=tk.HORIZONTAL,
                length=150,
                command=lambda v, idx=i: self._on_analog_out_change(idx, v)
            )
            scale.pack(side=tk.LEFT, padx=5)

            ttk.Label(row_frame, textvariable=var, width=6).pack(side=tk.LEFT)

    def _create_log_area(self, parent):
        """创建日志区域"""
        frame = ttk.LabelFrame(parent, text="日志", padding=10)
        frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 创建文本框和滚动条
        scroll = ttk.Scrollbar(frame)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.log_text = tk.Text(frame, height=10, width=30, yscrollcommand=scroll.set)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        scroll.config(command=self.log_text.yview)

        # 配置标签
        self.log_text.tag_config("INFO", foreground="black")
        self.log_text.tag_config("WARNING", foreground="orange")
        self.log_text.tag_config("ERROR", foreground="red")
        self.log_text.tag_config("SUCCESS", foreground="green")

    def _log(self, message: str, level: str = "INFO"):
        """添加日志"""
        self.log_text.insert(tk.END, f"[{level}] {message}\n", level)
        self.log_text.see(tk.END)

    # ==================== 事件处理 ====================

    def _on_connect(self):
        """连接按钮点击事件"""
        host = self.host_var.get().strip()
        port = self.port_var.get()

        if not host:
            messagebox.showerror("错误", "请输入 IP 地址")
            return

        # 在后台线程连接
        def connect_thread():
            try:
                success = self.controller.connect(host, port)
                if success:
                    self.root.after(0, lambda: self._log("连接成功", "SUCCESS"))
                else:
                    self.root.after(0, lambda: self._log("连接失败", "ERROR"))
            except Exception as e:
                self.root.after(0, lambda: self._log(f"连接异常: {e}", "ERROR"))

        threading.Thread(target=connect_thread, daemon=True).start()
        self._log(f"正在连接到 {host}:{port}...", "INFO")

    def _on_disconnect(self):
        """断开按钮点击事件"""
        self.controller.disconnect()
        self._log("已断开连接", "INFO")

    def _on_move(self):
        """执行移动"""
        if not self.controller.is_connected:
            messagebox.showwarning("警告", "未连接到机械臂")
            return

        try:
            acceleration = self.acceleration.get()
            velocity = self.velocity.get()

            if self.move_type.get() == "MOVEL":
                # 笛卡尔移动
                pose = Pose(
                    x=self.pose_x.get(),
                    y=self.pose_y.get(),
                    z=self.pose_z.get(),
                    rx=self.pose_rx.get(),
                    ry=self.pose_ry.get(),
                    rz=self.pose_rz.get()
                )

                time = self.move_time.get() if self.use_time.get() else 0.0
                success = self.controller.movel(pose, acceleration, velocity, time)

                if success:
                    self._log(f"MOVEL: {pose}", "INFO")
                else:
                    self._log("MOVEL 失败", "ERROR")

            else:
                # 关节移动
                joints = JointAngles(
                    j0=self.joint_j0.get(),
                    j1=self.joint_j1.get(),
                    j2=self.joint_j2.get(),
                    j3=self.joint_j3.get(),
                    j4=self.joint_j4.get(),
                    j5=self.joint_j5.get()
                )

                time = self.move_time.get() if self.use_time.get() else 0.0
                success = self.controller.movej(joints, acceleration, velocity, time)

                if success:
                    self._log(f"MOVEJ: {joints}", "INFO")
                else:
                    self._log("MOVEJ 失败", "ERROR")

        except ValueError as e:
            messagebox.showerror("错误", f"参数错误: {e}")

    def _on_stopj(self):
        """关节停止"""
        if self.controller.stopj():
            self._log("STOPJ: 已停止关节运动", "WARNING")

    def _on_stopl(self):
        """笛卡尔停止"""
        if self.controller.stopl():
            self._log("STOPL: 已停止笛卡尔运动", "WARNING")

    def _on_digital_out_change(self, pin: int):
        """数字输出变化"""
        if not self.controller.is_connected:
            return

        value = self.digital_outs[pin].get()
        self.controller.set_digital_out(pin, value)
        self._log(f"数字输出: D{pin} = {value}", "INFO")

    def _on_analog_out_change(self, pin: int, value: str):
        """模拟输出变化"""
        if not self.controller.is_connected:
            return

        try:
            val = float(value)
            self.controller.set_analog_out(pin, val)
            # self._log(f"模拟输出: A{pin} = {val:.3f}", "INFO")  # 太频繁了
        except ValueError:
            pass

    def _preset_home(self):
        """预设: Home 位置"""
        self.pose_x.set(0.3)
        self.pose_y.set(0.2)
        self.pose_z.set(0.4)
        self.pose_rx.set(0.0)
        self.pose_ry.set(3.14)
        self.pose_rz.set(0.0)
        self._log("加载预设: Home", "INFO")

    def _preset_zero(self):
        """预设: Zero 位置"""
        self.joint_j0.set(0.0)
        self.joint_j1.set(0.0)
        self.joint_j2.set(0.0)
        self.joint_j3.set(0.0)
        self.joint_j4.set(0.0)
        self.joint_j5.set(0.0)
        self._log("加载预设: Zero", "INFO")

    def _preset_above(self):
        """预设: 上方位置"""
        self.pose_x.set(0.0)
        self.pose_y.set(0.0)
        self.pose_z.set(0.5)
        self.pose_rx.set(0.0)
        self.pose_ry.set(3.14)
        self.pose_rz.set(0.0)
        self._log("加载预设: 上方", "INFO")

    def _on_status_changed(self, status: ConnectionStatus):
        """状态变化回调"""
        self.root.after(0, lambda: self._update_status(status))

    def _update_status(self, status: ConnectionStatus):
        """更新状态显示"""
        self.status_text.set(status.value)

        if status == ConnectionStatus.CONNECTED:
            self.connect_btn.config(state=tk.DISABLED)
            self.disconnect_btn.config(state=tk.NORMAL)
        else:
            self.connect_btn.config(state=tk.NORMAL)
            self.disconnect_btn.config(state=tk.DISABLED)

    def _on_error(self, error_msg: str):
        """错误回调"""
        self.root.after(0, lambda: self._log(error_msg, "ERROR"))

    def _setup_hotkeys(self):
        """设置快捷键"""
        # 注册默认快捷键（Ctrl+T 切换置顶）
        self.hotkey_manager.register_default_hotkeys(self._on_topmost_changed)
        self._log("快捷键已注册: Ctrl+T 切换窗口置顶", "INFO")

    def _on_topmost_changed(self, topmost: bool):
        """置顶状态变化回调"""
        # 更新标题栏
        self.topmost_indicator.update(topmost)

        # 更新状态显示
        if topmost:
            self.topmost_status_text.set("⬆ 置顶模式")
            self.topmost_label.config(foreground="red")
            self._log("窗口已切换到置顶模式", "SUCCESS")
        else:
            self.topmost_status_text.set("普通模式")
            self.topmost_label.config(foreground="purple")
            self._log("窗口已取消置顶", "INFO")


def main():
    """主函数"""
    root = tk.Tk()
    app = UR5ControlPanel(root)

    # 处理窗口关闭
    def on_closing():
        app.controller.disconnect()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)

    root.mainloop()


if __name__ == "__main__":
    main()
