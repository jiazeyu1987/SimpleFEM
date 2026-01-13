#!/usr/bin/env python3
"""分析新的缓存文件"""
import json

cache_file = r"D:\ProjectPackage\SimpleFEM\fem_refactor\external\export\roi_analysis_cache_20260112_173006_75e026e55668.jsonl"

target_frames = [25, 68, 117, 126, 167, 238]

print("=" * 100)
print("新缓存文件分析 (方案1+方案3)")
print("=" * 100)

with open(cache_file, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            data = json.loads(line.strip())
            if data.get('type') == 'frame':
                frame_idx = data.get('frame_index')
                if frame_idx in target_frames:
                    print(f"\n{'='*100}")
                    print(f"第 {frame_idx} 帧分析")
                    print(f"{'='*100}")

                    # ROI2 数据
                    roi2_gray = data.get('roi2_gray', 'N/A')
                    print(f"\n【ROI2 数据】")
                    print(f"  灰度值: {roi2_gray}")

                    # 阈值信息
                    threshold = data.get('threshold', {})
                    print(f"\n【阈值信息】")
                    print(f"  固定阈值: {threshold.get('fixed', 'N/A')}")
                    print(f"  使用阈值: {threshold.get('used', 'N/A')}")
                    print(f"  自适应窗口帧数: {threshold.get('adaptive_window_frames', 'N/A')}")
                    print(f"  背景均值: {threshold.get('calculated_bg_mean', 'N/A')}")
                    print(f"  保护激活: {threshold.get('protection_active', 'N/A')}")

                    # 判断是否超过阈值
                    used_threshold = threshold.get('used', 0)
                    if roi2_gray != 'N/A' and used_threshold:
                        is_above = roi2_gray >= used_threshold
                        print(f"\n  ⚡ ROI2 灰度值 {roi2_gray:.2f} {'≥' if is_above else '<'} 阈值 {used_threshold:.2f}")
                        if is_above:
                            print(f"  ✅ ROI2 超过阈值，应该能检测到")
                        else:
                            print(f"  ❌ ROI2 低于阈值，无法检测")

                    # ROI1 数据
                    roi1_gray = data.get('roi1_gray', 'N/A')
                    roi1_peaks = data.get('roi1_peaks', {'green': [], 'red': []})
                    print(f"\n【ROI1 数据】")
                    print(f"  ROI1 灰度值: {roi1_gray}")
                    print(f"  ROI1 绿色波峰: {roi1_peaks.get('green', [])}")
                    print(f"  ROI1 红色波峰: {roi1_peaks.get('red', [])}")

                    # 检测模式
                    detection = data.get('detection', {})
                    print(f"\n【检测模式】")
                    print(f"  模式: {detection.get('mode', 'N/A')}")
                    print(f"  混合检测启用: {detection.get('hybrid_enabled', 'N/A')}")
                    print(f"  ROI1 检测启用: {detection.get('roi1_enabled', 'N/A')}")

                    # 最终波峰
                    peaks = data.get('peaks', {})
                    print(f"\n【最终波峰】")
                    print(f"  绿色原始波峰: {peaks.get('green_raw', [])}")
                    print(f"  红色原始波峰: {peaks.get('red_raw', [])}")
                    print(f"  最终绿色波峰: {peaks.get('green', [])}")
                    print(f"  最终红色波峰: {peaks.get('red', [])}")

        except Exception as e:
            pass

print("\n" + "=" * 100)
print("总结")
print("=" * 100)
