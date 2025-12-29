#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
创建测试用的ROI1图片序列
生成roi1_000050.png 到 roi1_000060.png
"""

import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def create_test_sequence():
    """创建测试用的图片序列"""
    print("创建测试图片序列...")

    # 创建输出目录
    output_dir = "test_images"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 尝试加载字体
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    # 生成11张图片 (000050-000060)
    for i in range(50, 61):
        # 创建200x200的图片
        img = Image.new('RGB', (200, 200), color='white')
        draw = ImageDraw.Draw(img)

        # 绘制图片编号
        text = f"ROI1_{i:06d}"
        draw.text((10, 10), text, fill='black', font=font)

        # 绘制不同的ROI区域模式
        center_x, center_y = 100, 100

        # 根据序号创建不同的灰度模式
        gray_value = 50 + (i - 50) * 15  # 从50到200的灰度值
        gray_rgb = (gray_value, gray_value, gray_value)

        # ROI2区域（变化）
        roi2_size = 30 + (i % 5) * 5
        draw.rectangle([center_x - roi2_size, center_y - roi2_size,
                       center_x + roi2_size, center_y + roi2_size],
                      outline='red', width=2)

        # ROI3区域（固定）
        roi3_size = 50
        # 先绘制实线框
        draw.rectangle([center_x - roi3_size, center_y - roi3_size,
                       center_x + roi3_size, center_y + roi3_size],
                      outline='blue', width=2)

        # 添加一些随机纹理
        np.random.seed(i)
        for _ in range(20):
            x = np.random.randint(0, 200)
            y = np.random.randint(0, 200)
            size = np.random.randint(2, 8)
            color_val = np.random.randint(30, 220)
            color = (color_val, color_val, color_val)
            draw.ellipse([x, y, x+size, y+size], fill=color)

        # 保存图片
        filename = f"roi1_{i:06d}.png"
        filepath = os.path.join(output_dir, filename)
        img.save(filepath)

        print(f"  创建: {filename}")

    print(f"\n[OK] 成功创建11张测试图片，保存在 {output_dir}/ 目录")
    print("   可以使用这些图片测试键盘导航功能:")
    print("   - 导入 roi1_000054.png")
    print("   - 按 D 或 -> : 切换到 roi1_000055.png")
    print("   - 按 A 或 <- : 切换到 roi1_000053.png")

if __name__ == "__main__":
    create_test_sequence()