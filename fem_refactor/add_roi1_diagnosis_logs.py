#!/usr/bin/env python3
"""添加ROI1诊断日志到detection_pipeline.py"""
import os

file_path = r"D:\ProjectPackage\SimpleFEM\fem_refactor\detection_pipeline.py"

# 读取原文件
with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 查找并替换目标代码段
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
        return []

    # 合并ROI1检测到的所有波峰（不区分颜色）
    roi1_all_peaks = roi1_green_peaks_raw + roi1_red_peaks_raw
'''

# 查找替换位置（第56行开始，到第74行结束）
start_idx = None
end_idx = None

for i, line in enumerate(lines):
    if i == 55 and 'buffer_start_frame_index = int(config.get("buffer_start_frame_index", 1))' in line:
        start_idx = i
    if start_idx is not None and i > start_idx:
        if '# 合并ROI1检测到的所有波峰（不区分颜色）' in line:
            end_idx = i
            break

if start_idx is None or end_idx is None:
    print(f"错误: 无法找到目标代码段")
    print(f"start_idx={start_idx}, end_idx={end_idx}")
else:
    # 替换代码
    new_lines = lines[:start_idx] + [new_code] + lines[end_idx:]

    # 写入备份文件
    backup_path = file_path + ".backup"
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print(f"已创建备份: {backup_path}")

    # 写入新文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

    print(f"✅ 成功添加ROI1诊断日志到 {file_path}")
    print(f"修改了第{start_idx+1}行到第{end_idx}行")
