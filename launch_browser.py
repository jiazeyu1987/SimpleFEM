#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ROI Cache Browser 快速启动器
自动打开最新的缓存文件
"""

import glob
import os
import sys
from pathlib import Path

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def find_latest_cache_file():
    """查找最新的 JSONL 缓存文件"""
    # 搜索 export 目录
    export_dir = Path('export')
    if not export_dir.exists():
        # 如果 export 目录不存在，搜索当前目录
        cache_files = list(Path('.').glob('roi_analysis_cache_*.jsonl'))
    else:
        cache_files = list(export_dir.glob('roi_analysis_cache_*.jsonl'))

    if not cache_files:
        print("错误: 未找到任何 roi_analysis_cache_*.jsonl 文件")
        print("\n请确保以下位置之一存在缓存文件:")
        print("  - ./export/roi_analysis_cache_*.jsonl")
        print("  - ./roi_analysis_cache_*.jsonl")
        return None

    # 按修改时间排序，获取最新的
    latest = max(cache_files, key=lambda f: f.stat().st_mtime)
    return latest

def main():
    print("="*80)
    print("ROI Analysis Cache Browser - 快速启动")
    print("="*80)

    # 查找最新文件
    cache_file = find_latest_cache_file()
    if not cache_file:
        input("\n按回车键退出...")
        return

    print(f"\n找到最新的缓存文件:")
    print(f"  文件名: {cache_file.name}")
    print(f"  位置: {cache_file}")
    print(f"  大小: {cache_file.stat().st_size / 1024:.2f} KB")

    # 导入并启动浏览器
    try:
        from roi_cache_browser import CacheBrowserGUI
        import tkinter as tk

        print(f"\n正在启动浏览器...")
        root = tk.Tk()
        app = CacheBrowserGUI(root)

        # 自动加载文件
        print(f"正在加载数据...")
        app.load_file(str(cache_file))

        print(f"\n[OK] 浏览器已启动！")
        print(f"[OK] 数据已加载: {cache_file.name}")
        print(f"\n提示:")
        print(f"  - 悬停在字段名上查看说明")
        print(f"  - 使用导航按钮浏览帧数据")
        print(f"  - 切换标签页查看不同视图")

        root.mainloop()

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        input("\n按回车键退出...")

if __name__ == '__main__':
    main()
