#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
快捷键管理类

提供全局快捷键功能，用于窗口控制等操作。
"""

import tkinter as tk
from tkinter import ttk
from typing import Callable, Dict, Optional, List
import logging


logger = logging.getLogger(__name__)


class HotKeyManager:
    """
    快捷键管理类

    功能：
    - 管理窗口快捷键绑定
    - 支持窗口置顶切换 (Ctrl+T)
    - 可扩展其他快捷键功能
    """

    def __init__(self, root: tk.Tk):
        """
        初始化快捷键管理器

        Args:
            root: Tkinter 根窗口
        """
        self.root = root

        # 快捷键回调字典
        self._callbacks: Dict[str, Callable] = {}

        # 状态标志
        self._topmost = False

        logger.info("快捷键管理器初始化")

    @property
    def is_topmost(self) -> bool:
        """是否窗口置顶"""
        return self._topmost

    def register_hotkey(self, key_sequence: str, callback: Callable, description: str = ""):
        """
        注册快捷键

        Args:
            key_sequence: 快捷键序列（如 '<Control-t>', '<F1>'）
            callback: 回调函数
            description: 快捷键描述
        """
        try:
            self.root.bind(key_sequence, callback)
            self._callbacks[key_sequence] = callback

            logger.debug(f"注册快捷键: {key_sequence} - {description}")

        except Exception as e:
            logger.error(f"注册快捷键失败 {key_sequence}: {e}")

    def unregister_hotkey(self, key_sequence: str):
        """
        注销快捷键

        Args:
            key_sequence: 快捷键序列
        """
        try:
            self.root.unbind(key_sequence)
            if key_sequence in self._callbacks:
                del self._callbacks[key_sequence]

            logger.debug(f"注销快捷键: {key_sequence}")

        except Exception as e:
            logger.error(f"注销快捷键失败 {key_sequence}: {e}")

    def toggle_topmost(self, event=None):
        """
        切换窗口置顶状态 (Ctrl+T)

        Args:
            event: Tkinter 事件对象（可选）
        """
        self._topmost = not self._topmost
        self.root.attributes('-topmost', self._topmost)

        status = "置顶" if self._topmost else "取消置顶"
        logger.info(f"窗口已{status}")

        # 返回 'break' 阻止事件继续传播
        return 'break'

    def set_topmost(self, topmost: bool):
        """
        设置窗口置顶状态

        Args:
            topmost: True=置顶, False=取消置顶
        """
        self._topmost = topmost
        self.root.attributes('-topmost', topmost)

        status = "置顶" if topmost else "取消置顶"
        logger.info(f"窗口已{status}")

    def register_default_hotkeys(self, status_callback: Optional[Callable[[bool], None]] = None):
        """
        注册默认快捷键

        Args:
            status_callback: 状态变化回调函数（接收置顶状态参数）
        """
        # Ctrl+T: 切换窗口置顶
        def on_toggle_topmost(event=None):
            self.toggle_topmost(event)
            if status_callback:
                status_callback(self._topmost)
            return 'break'

        self.register_hotkey('<Control-t>', on_toggle_topmost, "切换窗口置顶")
        self.register_hotkey('<Control-T>', on_toggle_topmost, "切换窗口置顶(大写)")

        logger.info("默认快捷键已注册")

    def get_hotkey_list(self) -> List[Dict[str, str]]:
        """
        获取已注册的快捷键列表

        Returns:
            快捷键信息列表
        """
        # 这里返回常用快捷键说明
        return [
            {"key": "Ctrl+T", "description": "切换窗口置顶/取消置顶"},
        ]


class TopmostIndicator:
    """
    窗口置顶指示器

    在窗口标题栏显示置顶状态
    """

    def __init__(self, root: tk.Tk, original_title: str):
        """
        初始化置顶指示器

        Args:
            root: Tkinter 根窗口
            original_title: 原始窗口标题
        """
        self.root = root
        self.original_title = original_title
        self.is_topmost = False

    def update(self, topmost: bool):
        """
        更新置顶状态

        Args:
            topmost: 是否置顶
        """
        self.is_topmost = topmost

        if topmost:
            self.root.title(f"⬆ {self.original_title}")
        else:
            self.root.title(self.original_title)


if __name__ == "__main__":
    # 测试代码
    root = tk.Tk()
    root.title("快捷键测试")
    root.geometry("400x300")

    # 创建快捷键管理器
    hotkey_manager = HotKeyManager(root)

    # 创建指示器
    indicator = TopmostIndicator(root, "快捷键测试")

    # 状态标签
    status_label = ttk.Label(root, text="按 Ctrl+T 切换窗口置顶", font=("Arial", 14))
    status_label.pack(pady=50)

    def on_topmost_change(topmost: bool):
        status = "置顶" if topmost else "普通"
        status_label.config(text=f"窗口模式: {status}\n按 Ctrl+T 切换")
        indicator.update(topmost)

    # 注册默认快捷键
    hotkey_manager.register_default_hotkeys(on_topmost_change)

    # 显示快捷键列表
    info_frame = ttk.Frame(root)
    info_frame.pack(pady=20)

    ttk.Label(info_frame, text="可用快捷键:", font=("Arial", 12, "bold")).pack()

    for hotkey in hotkey_manager.get_hotkey_list():
        ttk.Label(
            info_frame,
            text=f"{hotkey['key']}: {hotkey['description']}",
            font=("Arial", 10)
        ).pack(pady=5)

    root.mainloop()
