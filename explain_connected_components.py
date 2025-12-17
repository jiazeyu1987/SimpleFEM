#!/usr/bin/env python3
"""
详细解释连通域(Connected Components)概念和判定逻辑
"""

import sys
import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
sys.path.append('.')

def explain_connected_components():
    """详细解释连通域概念"""
    print("=== 连通域(Connected Components)概念解释 ===")

    print("\n1. 什么是连通域？")
    print("   连通域是指在图像中，相互连接的像素点的集合。")
    print("   '连接'意味着可以通过特定规则从一个像素移动到另一个像素。")
    print("")

    print("2. 连通性定义:")
    print("   a) 4-连通(4-connectivity): 只能上下左右移动")
    print("      ✘")
    print("      ○")
    print("      ✘")
    print("      ○")
    print("      ○")
    print("      ✘")
    print("")
    print("   b) 8-连通(8-connectivity): 可以8个方向移动（包括对角线）")
    print("      ✘ ✘ ✘")
    print("      ○ ○ ○")
    print("      ○ ○ ○")
    print("      ✘ ✘ ✘")
    print("      ○ ○ ○")
    print("      ✘ ✘ ✘")

    print("\n3. 连通域判定规则:")
    print("   - 连通域内的任意两个像素点都存在一条路径")
    print("   - 路径上的所有像素都属于同一个连通域")
    print("   - 路径只能通过'连接'关系移动")
    print("   - 不同连通域之间的像素无法通过'连接'关系相互到达")

    print("\n4. 在代码中的实现:")

    # 创建示例图像来演示
    print("\n   创建示例图像...")

    # 创建一个100x100的二值图像
    binary_image = np.zeros((100, 100), dtype=np.uint8)

    # 创建几个分离的连通域
    # 连通域1: 左上角的矩形
    binary_image[10:30, 10:40] = 255

    # 连通域2: 右上角的L形
    binary_image[10:30, 60:80] = 255
    binary_image[25:30, 60:70] = 255

    # 连通域3: 底部的圆形区域
    for y in range(70, 90):
        for x in range(70, 90):
            if ((x-79)**2 + (y-79)**2) <= 100:  # 圆形
                binary_image[y, x] = 255

    # 连通域4: 右下角的点（太小会被min_component_size过滤）
    binary_image[85, 85] = 255

    print("   创建了4个分离的连通域:")
    print("   - 矩形 (左上)")
    print("   - L形 (右上)")
    print("   - 圆形 (底部)")
    print("   - 单点 (右下)")

    # 保存示例图像
    cv2.imwrite("example_binary_image.png", binary_image)
    print("   示例图像已保存: example_binary_image.png")

    # 使用OpenCV进行连通域分析
    print("\n5. OpenCV连通域分析过程:")

    # 4-连通性分析
    num_labels_4, labels_4 = cv2.connectedComponents(binary_image, connectivity=4)
    print(f"   4-连通性结果: {num_labels_4-1} 个连通域 (标签1-{num_labels_4-1})")

    # 8-连通性分析
    num_labels_8, labels_8 = cv2.connectedComponents(binary_image, connectivity=8)
    print(f"   8-连通性结果: {num_labels_8-1} 个连通域 (标签1-{num_labels_8-1})")

    # 详细分析4-连通性的结果
    print(f"\n6. 4-连通性详细分析:")
    for label in range(1, num_labels_4):
        # 获取这个连通域的像素坐标
        component_mask = (labels_4 == label)
        pixels = np.sum(component_mask)

        if pixels == 0:
            continue

        # 找到边界框
        coords = np.where(component_mask)
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()

        print(f"   连通域 {label}:")
        print(f"     像素数量: {pixels}")
        print(f"     边界框: ({x_min:2d}, {y_min:2d}) 到 ({x_max:2d}, {y_max:2d})")
        print(f"     尺寸: {x_max-x_min+1} x {y_max-y_min+1}")

    # 可视化连通域
    print(f"\n7. 连通域可视化:")

    # 创建彩色可视化
    colored_4 = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
    colors = [
        (255, 0, 0),    # 红色 - 连通域1
        (0, 255, 0),    # 绿色 - 连通域2
        (0, 0, 255),    # 蓝色 - 连通域3
        (255, 255, 0)    # 黄色 - 连通域4
    ]

    for label in range(1, min(num_labels_4, len(colors) + 1)):
        mask = (labels_4 == label)
        colored_4[mask] = colors[label-1]

        # 添加标签文字
        coords = np.where(mask)
        if len(coords[0]) > 0:
            y_center = int(np.mean(coords[0]))
            x_center = int(np.mean(coords[1]))
            cv2.putText(colored_4, str(label), (x_center-5, y_center+5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imwrite("connected_components_4way.png", colored_4)
    print("   4-连通域可视化: connected_components_4way.png")

    # 展示连通路径的概念
    print(f"\n8. 连通路径演示:")

    # 创建一个更复杂的例子来展示连通路径
    path_example = np.zeros((50, 50), dtype=np.uint8)

    # 创建一个蛇形连通域
    path_example[10, 10] = 255  # 起点
    path_example[11, 10] = 255
    path_example[12, 10] = 255
    path_example[13, 10] = 255
    path_example[13, 11] = 255
    path_example[13, 12] = 255
    path_example[13, 13] = 255
    path_example[13, 14] = 255
    path_example[12, 14] = 255
    path_example[11, 14] = 255
    path_example[10, 14] = 255
    path_example[10, 13] = 255
    path_example[10, 12] = 255
    path_example[10, 11] = 255

    path_example[25, 25] = 255  # 另一个独立的连通域

    cv2.imwrite("path_example.png", path_example)
    print("   连通路径示例: path_example.png")
    print("   第一个连通域是一条蛇形路径，所有像素都是连通的")
    print("   第二个连通域是一个独立的点")

    # 分析路径示例
    num_labels_path, labels_path = cv2.connectedComponents(path_example, connectivity=4)
    print(f"   路径示例连通域: {num_labels_path-1} 个")

    # 8连通性 vs 4-连通性的差异演示
    print(f"\n9. 4-连通 vs 8-连通的差异:")

    diagonal_test = np.zeros((5, 5), dtype=np.uint8)
    diagonal_test[2, 2] = 255  # 中心点
    diagonal_test[1, 1] = 255  # 对角线连接
    diagonal_test[3, 3] = 255  # 对角线连接

    cv2.imwrite("diagonal_test.png", diagonal_test)
    print("   对角线测试: diagonal_test.png")

    # 4-连通性分析
    num_labels_4d, labels_4d = cv2.connectedComponents(diagonal_test, connectivity=4)

    # 8-连通性分析
    num_labels_8d, labels_8d = cv2.connectedComponents(diagonal_test, connectivity=8)

    print(f"   4-连通性: {num_labels_4d-1} 个连通域 (对角线不连通)")
    print(f"   8-连通性: {num_labels_8d-1} 个连通域 (对角线连通)")

    print(f"\n10. auto_vein_detector.py中的连通域使用:")
    print(f"   connectivity = {4}  (4-连通性)")
    print(f"   连通域用途:")
    print(f"   - 分离不同的血管或特征区域")
    print(f"   - 过除噪声点（通过min_component_size）")
    print(f"   - 选择ROI中心所在的区域")
    print(f"   - 实现精确的特征提取")

    print(f"\n11. 连通域在图像处理中的重要性:")
    print(f"   ✓ 特征分离: 将不同的对象或区域分开")
    print(f"   ✓ 噪声过滤: 小的孤立点通常是噪声")
    f"   ✓ 形状分析: 连通域的形状提供对象信息")
    print(f"   ✓ 区域跟踪: 跟踪特定连通域的移动")
    print(f"   ✓ 语义分割: 将图像分割成有意义的区域")

    print(f"\n=== 连通域概念解释完成 ===")
    print(f"关键要点:")
    print(f"  • 连通域 = 相互连接的像素集合")
    print(f"  • 连接性定义 = 像素间的移动规则 (4-连通或8-连通)")
    print(f"  • 判定方法 = OpenCV的connectedComponents函数")
    print(f"  • 实际应用 = 过滤、分割、特征提取")

if __name__ == "__main__":
    explain_connected_components()