#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UR5 机械臂控制面板 - 主入口

这是 UR5 控制面板的主入口文件。
直接运行此文件即可启动控制面板。

使用方法:
    python ur5_main.py

作者: SimpleFEM Team
日期: 2025-12-30
"""

import sys
import logging
from tkinter import Tk

from ur5_control_panel import UR5ControlPanel


def setup_logging():
    """配置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('ur5_control.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )


def main():
    """主函数"""
    # 配置日志
    setup_logging()

    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("UR5 机械臂控制面板启动")
    logger.info("=" * 60)

    # 创建主窗口
    root = Tk()

    # 创建控制面板
    app = UR5ControlPanel(root)

    # 处理窗口关闭事件
    def on_closing():
        """窗口关闭处理"""
        logger.info("正在关闭控制面板...")
        app.controller.disconnect()
        logger.info("已断开连接")
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)

    # 启动主循环
    logger.info("控制面板已启动")
    root.mainloop()

    logger.info("控制面板已关闭")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"程序异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
