#!/usr/bin/env python3
"""直接修改detection_pipeline.py添加ROI1诊断日志"""
import re

file_path = r"D:\ProjectPackage\SimpleFEM\fem_refactor\detection_pipeline.py"

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 定义要替换的代码段
old_code = '''    buffer_start_frame_index = int(config.get("buffer_start_frame_index", 1))

    # 1. 使用ROI1数据进行波峰检测（ROI1独立阈值）
    try:
        roi1_green_peaks_raw, roi1_red_peaks_raw = detect_peaks(
            roi1_curve,
            threshold=config['roi1_threshold'],
            marginFrames=config['margin_frames'],
            silenceFrames=config['silence_frames'],
            avgFrames=config['pre_post_avg_frames'],
            differenceThreshold=999.0,  # 设为很大的值，让ROI1只检测波峰，不做颜色分类
        )
    except Exception as e:
        logging.error(f"[混合检测] ROI1波峰检测失败: {e}")
        print(f"[混合检测] ROI1波峰检测失败: {e}")
        return []'''

new_code = '''    buffer_start_frame_index = int(config.get("buffer_start_frame_index", 1))

    # ROI1曲线统计信息
    current_frame_index = config.get('frame_index', 0)
    roi1_threshold = config['roi1_threshold']
    roi1_curve_len = len(roi1_curve)

    if roi1_curve_len > 0:
        roi1_max = max(roi1_curve)
        roi1_max_idx = roi1_curve.index(roi1_max)
        roi1_above_threshold_count = sum(1 for v in roi1_curve if v >= roi1_threshold)
        roi1_first_10 = [f"{v:.1f}" for v in roi1_curve[:10]]
        roi1_last_5 = [f"{v:.1f}" for v in roi1_curve[-5:]] if len(roi1_curve) >= 5 else []

        logging.info(f"[ROI1诊断] 帧{current_frame_index} "
                    f"曲线长度={roi1_curve_len}, 阈值={roi1_threshold:.1f}, "
                    f"最大值={roi1_max:.2f}(索引{roi1_max_idx}), "
                    f"超过阈值帧数={roi1_above_threshold_count}/{roi1_curve_len}")
        logging.info(f"[ROI1诊断] 帧{current_frame_index} "
                    f"前10帧值: [{', '.join(roi1_first_10)}{('...' if len(roi1_curve) > 10 else '')}] "
                    f"后5帧值: [{', '.join(roi1_last_5)}]")
    else:
        logging.warning(f"[ROI1诊断] 帧{current_frame_index} ROI1曲线为空")

    # 1. 使用ROI1数据进行波峰检测（ROI1独立阈值）
    try:
        roi1_green_peaks_raw, roi1_red_peaks_raw = detect_peaks(
            roi1_curve,
            threshold=roi1_threshold,
            marginFrames=config['margin_frames'],
            silenceFrames=config['silence_frames'],
            avgFrames=config['pre_post_avg_frames'],
            differenceThreshold=999.0,  # 设为很大的值，让ROI1只检测波峰，不做颜色分类
        )

        total_peaks_detected = len(roi1_green_peaks_raw) + len(roi1_red_peaks_raw)
        logging.info(f"[ROI1诊断] 帧{current_frame_index} "
                    f"detect_peaks返回 {total_peaks_detected} 个波峰 (绿色:{len(roi1_green_peaks_raw)}, 红色:{len(roi1_red_peaks_raw)})")

        # 如果没有检测到波峰，输出诊断信息
        if total_peaks_detected == 0:
            if roi1_above_threshold_count == 0:
                logging.warning(f"[ROI1诊断] 帧{current_frame_index} "
                              f"失败原因: ROI1最大值{roi1_max:.2f} < 阈值{roi1_threshold:.1f}，没有任何帧超过阈值")
            else:
                logging.warning(f"[ROI1诊断] 帧{current_frame_index} "
                              f"失败原因: 有{roi1_above_threshold_count}帧超过阈值，但未形成波峰。"
                              f"可能原因: 1)silence_frames(前{config['silence_frames']}+后{config['silence_frames']}帧安静) "
                              f"2)margin_frames({config['margin_frames']}帧间隔) 3)frame_diff>15")
    except Exception as e:
        logging.error(f"[混合检测] ROI1波峰检测失败: {e}")
        print(f"[混合检测] ROI1波峰检测失败: {e}")
        return []'''

# 执行替换
if old_code in content:
    content = content.replace(old_code, new_code)

    # 写入备份
    with open(file_path + ".backup2", 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ 成功创建备份: detection_pipeline.py.backup2")

    # 写入原文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ 成功添加ROI1诊断日志")
else:
    print("❌ 错误: 无法找到目标代码段")
    print("文件可能已经被修改，请手动检查")
