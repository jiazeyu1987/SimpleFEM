#!/usr/bin/env python3
"""
简化的连通域概念解释
"""

import sys
import os
import numpy as np
import cv2
sys.path.append('.')

def simple_explanation():
    """简化解释连通域概念"""
    print("=== 连通域(Connected Components)概念解释 ===")

    print("\n1. 连通域定义:")
    print("   连通域 = 图像中相互连接的像素集合")
    print("   连接规则:")
    print("   - 4-连通: 上下左右 4个方向")
    print("   - 8-连通: 8个方向(包括对角线)")

    print("\n2. 判定逻辑:")
    print("   使用OpenCV的connectedComponents函数:")
    print("   - 标签0: 背景")
    print("   - 标签1,2,3...: 不同的连通域")

    print("\n3. auto_vein_detector.py中的使用:")

    # 检查配置
    config_path = "vein_detection_config.json"
    if os.path.exists(config_path):
        import json
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        print(f"   阈值范围: {config.get('vein_detection', {}).get('threshold_min', 'N/A')}-{config.get('vein_detection', {}).get('threshold_max', 'N/A')}")
        print(f"   连通性: {config.get('vein_detection', {}).get('connectivity', 'N/A')}")
        print(f"   最小组件大小: {config.get('vein_detection', {}).get('min_component_size', 'N/A')}")

    print("\n4. 连通域在代码中的应用:")

    # 创建简单示例
    print("\n   创建示例图像...")
    test_image = np.zeros((10, 10), dtype=np.uint8)

    # 创建几个连通域
    test_image[1, 1] = 255    # 连通域1 - 单点
    test_image[1, 2] = 255    # 连通域1 - 相邻点
    test_image[1, 3] = 255    # 连通域1 - 连续点
    test_image[5, 5] = 255    # 连通域2 - 独立点
    test_image[7, 7] = 255    # 连通域3 - 独立点

    cv2.imwrite("simple_example.png", test_image)
    print(f"   示例图像保存: simple_example.png")

    # 分析连通域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        test_image, connectivity=4, ltype=cv2.CV_32S
    )

    print(f"   分析结果:")
    print(f"   - 总标签数: {num_labels} (包括背景)")
    print(f"   - 连通域数: {num_labels-1}")

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        cx = int(centroids[i][0])
        cy = int(centroids[i][1])

        print(f"   连通域{i}: 面积={area:2d}, 位置=({x:2d},{y:2d}), 尺寸=({w}x{h}), 中心=({cx:2d},{cy:2d})")

    print(f"\n5. 中心连通域选择逻辑:")
    print(f"   步骤1: 获取ROI中心点坐标")
    print(f"   步骤2: 检查中心点是否在某个连通域内")
    print(f"   步骤3: 如果在，返回该连通域")
    print(f"   步骤4: 如果不在，计算所有连通域的质心")
    print(f"   步骤5: 选择距离中心点最近的连通域")
    print(f"   步骤6: 提取该连通域作为mask")

    print(f"\n6. 最小组件过滤:")
    print(f"   配置参数: min_component_size")
    print(f"   过滤逻辑: 面积 < min_component_size 的连通域被过滤")
    print(f"   目的: 消除噪声点和小的不相关区域")

    print(f"\n7. 在静脉检测中的作用:")
    print(f"   - 分离不同的血管特征")
    print(f"   - 精确选择ROI中心的区域")
    f"   - 过除背景噪声和小干扰")
    print(f"   - 实现基于连通域的特征提取")

    print(f"\n=== 总结 ===")
    print(f"连通域 = 图像分析的基础概念")
    print(f"判定 = 像素间的相邻关系(4/8连通)")
    print(f"应用 = OpenCV算法 + 配置参数过滤")
    print(f"目的 = 特征分离、噪声过滤、精确提取")

if __name__ == "__main__":
    simple_explanation()