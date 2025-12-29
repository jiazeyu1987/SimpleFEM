#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SimpleFEM ROI Analysis Cache Analyzer

独立的Python程序，用于读取和分析roi_analysis_cache_*.jsonl文件。

Usage:
    python analyze_cache.py <cache_file.jsonl> [options]

Options:
    --summary              显示基本统计摘要
    --peaks                详细分析波峰检测
    --intersection         分析绿线交点分布
    --waveform             绘制波形图
    --export-csv           导出到CSV文件
    --filter-frames X-Y    只分析指定帧范围 (例如: 100-200)
    --help                 显示帮助信息

Examples:
    python analyze_cache.py export/roi_analysis_cache_session_20251225_180811.jsonl --summary
    python analyze_cache.py export/roi_analysis_cache_*.jsonl --peaks --waveform
    python analyze_cache.py cache.jsonl --filter-frames 100-200 --export-csv
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.gridspec import GridSpec

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False


class CacheAnalyzer:
    """ROI分析缓存分析器"""

    def __init__(self, cache_file: str):
        self.cache_file = Path(cache_file)
        self.frames: List[Dict[str, Any]] = []
        self.meta: Optional[Dict[str, Any]] = None
        self.session_end: Optional[Dict[str, Any]] = None

        if not self.cache_file.exists():
            raise FileNotFoundError(f"缓存文件不存在: {cache_file}")

        self._load_cache()

    def _load_cache(self):
        """加载JSONL缓存文件"""
        print(f"正在加载缓存文件: {self.cache_file}")

        with open(self.cache_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    record_type = data.get('type', '')

                    if record_type == 'meta':
                        self.meta = data
                    elif record_type == 'frame':
                        self.frames.append(data)
                    elif record_type == 'session_end':
                        self.session_end = data
                except json.JSONDecodeError as e:
                    print(f"警告: 第{line_num}行JSON解析失败: {e}")

        print(f"加载完成: {len(self.frames)} 帧")

        if self.meta:
            print(f"会话ID: {self.meta.get('session_id', 'N/A')}")
            print(f"处理模式: {self.meta.get('processing_mode', 'N/A')}")
            print(f"创建时间: {self.meta.get('created_at', 'N/A')}")

    def get_summary(self) -> Dict[str, Any]:
        """获取基本统计摘要"""
        if not self.frames:
            return {}

        # 适配实际的数据格式
        roi2_values = []
        roi3_values = []

        for f in self.frames:
            # ROI2 灰度值
            if 'roi2_gray' in f:
                roi2_values.append(f['roi2_gray'])

            # ROI3 灰度值 (在 roi3.gray 字段中)
            if 'roi3' in f and isinstance(f['roi3'], dict):
                roi3_gray = f['roi3'].get('gray')
                if roi3_gray is not None:
                    roi3_values.append(roi3_gray)

        # 统计波峰 (从 peaks 字段中)
        total_green_peaks = 0
        total_red_peaks = 0
        frames_with_peaks = 0

        for f in self.frames:
            if 'peaks' in f and isinstance(f['peaks'], dict):
                peaks = f['peaks']
                green_count = len(peaks.get('green', []))
                red_count = len(peaks.get('red', []))
                total_green_peaks += green_count
                total_red_peaks += red_count
                if green_count > 0 or red_count > 0:
                    frames_with_peaks += 1

        # 统计交点
        valid_intersections = []
        for f in self.frames:
            if 'intersection' in f and isinstance(f['intersection'], dict):
                used = f['intersection'].get('used')
                if used and isinstance(used, list) and len(used) >= 2:
                    valid_intersections.append({'x': used[0], 'y': used[1]})

        summary = {
            'total_frames': len(self.frames),
            'roi2_stats': {
                'count': len(roi2_values),
                'mean': np.mean(roi2_values) if roi2_values else 0,
                'std': np.std(roi2_values) if roi2_values else 0,
                'min': np.min(roi2_values) if roi2_values else 0,
                'max': np.max(roi2_values) if roi2_values else 0,
            },
            'roi3_stats': {
                'count': len(roi3_values),
                'mean': np.mean(roi3_values) if roi3_values else 0,
                'std': np.std(roi3_values) if roi3_values else 0,
                'min': np.min(roi3_values) if roi3_values else 0,
                'max': np.max(roi3_values) if roi3_values else 0,
            },
            'peak_stats': {
                'frames_with_peaks': frames_with_peaks,
                'green_peaks': total_green_peaks,
                'red_peaks': total_red_peaks,
                'peak_rate': frames_with_peaks / len(self.frames) if self.frames else 0,
            },
            'intersection_stats': {
                'valid_count': len(valid_intersections),
                'invalid_count': len(self.frames) - len(valid_intersections),
            }
        }

        if valid_intersections:
            x_coords = [i['x'] for i in valid_intersections]
            y_coords = [i['y'] for i in valid_intersections]
            summary['intersection_stats'].update({
                'x_mean': np.mean(x_coords),
                'x_std': np.std(x_coords),
                'y_mean': np.mean(y_coords),
                'y_std': np.std(y_coords),
                'x_range': (np.min(x_coords), np.max(x_coords)),
                'y_range': (np.min(y_coords), np.max(y_coords)),
            })

        return summary

    def print_summary(self):
        """打印统计摘要"""
        summary = self.get_summary()

        print("\n" + "="*80)
        print("ROI 分析缓存统计摘要")
        print("="*80)

        print(f"\n总帧数: {summary['total_frames']}")
        print(f"有效ROI2数据: {summary['roi2_stats']['count']} 帧")
        print(f"有效ROI3数据: {summary['roi3_stats']['count']} 帧")

        print("\n" + "-"*80)
        print("ROI2 灰度值统计:")
        print(f"  均值: {summary['roi2_stats']['mean']:.2f}")
        print(f"  标准差: {summary['roi2_stats']['std']:.2f}")
        print(f"  范围: [{summary['roi2_stats']['min']:.2f}, {summary['roi2_stats']['max']:.2f}]")

        if summary['roi3_stats']['count'] > 0:
            print("\nROI3 灰度值统计:")
            print(f"  均值: {summary['roi3_stats']['mean']:.2f}")
            print(f"  标准差: {summary['roi3_stats']['std']:.2f}")
            print(f"  范围: [{summary['roi3_stats']['min']:.2f}, {summary['roi3_stats']['max']:.2f}]")

        print("\n" + "-"*80)
        print("波峰检测统计:")
        print(f"  检测到波峰的帧数: {summary['peak_stats']['frames_with_peaks']}")
        print(f"  绿色波峰总数: {summary['peak_stats']['green_peaks']}")
        print(f"  红色波峰总数: {summary['peak_stats']['red_peaks']}")
        print(f"  总波峰数: {summary['peak_stats']['green_peaks'] + summary['peak_stats']['red_peaks']}")
        print(f"  波峰检测率: {summary['peak_stats']['peak_rate']*100:.2f}%")

        print("\n" + "-"*80)
        print("绿线交点统计:")
        print(f"  有效交点数量: {summary['intersection_stats']['valid_count']}")
        print(f"  无效交点数量: {summary['intersection_stats']['invalid_count']}")

        if summary['intersection_stats']['valid_count'] > 0:
            print(f"\n交点X坐标:")
            print(f"  均值: {summary['intersection_stats']['x_mean']:.2f}")
            print(f"  标准差: {summary['intersection_stats']['x_std']:.2f}")
            print(f"  范围: [{summary['intersection_stats']['x_range'][0]:.2f}, {summary['intersection_stats']['x_range'][1]:.2f}]")
            print(f"\n交点Y坐标:")
            print(f"  均值: {summary['intersection_stats']['y_mean']:.2f}")
            print(f"  标准差: {summary['intersection_stats']['y_std']:.2f}")
            print(f"  范围: [{summary['intersection_stats']['y_range'][0]:.2f}, {summary['intersection_stats']['y_range'][1]:.2f}]")

        print("="*80 + "\n")

    def analyze_peaks(self):
        """详细分析波峰检测"""
        print("\n" + "="*80)
        print("波峰检测详细分析")
        print("="*80)

        peaks_by_frame = []
        all_green_peaks = []
        all_red_peaks = []

        for frame in self.frames:
            frame_idx = frame.get('frame_index', -1)
            if 'peaks' in frame and isinstance(frame['peaks'], dict):
                peaks = frame['peaks']
                green = peaks.get('green', [])
                red = peaks.get('red', [])

                if green or red:
                    peaks_by_frame.append({
                        'frame_index': frame_idx,
                        'green_count': len(green),
                        'red_count': len(red),
                        'green_peaks': green,
                        'red_peaks': red,
                    })

                all_green_peaks.extend([(frame_idx, g) for g in green])
                all_red_peaks.extend([(frame_idx, r) for r in red])

        total_peaks = len(all_green_peaks) + len(all_red_peaks)

        if total_peaks == 0:
            print("\n未检测到任何波峰")
            return

        print(f"\n总共检测到 {total_peaks} 个波峰")
        print(f"绿色波峰: {len(all_green_peaks)} 个")
        print(f"红色波峰: {len(all_red_peaks)} 个")
        print(f"检测到波峰的帧数: {len(peaks_by_frame)}")

        # 按颜色统计
        if all_green_peaks:
            frame_indices = [p[0] for p in all_green_peaks]
            print(f"\n绿色波峰分布:")
            print(f"  首个波峰: 帧索引 {min(frame_indices)}")
            print(f"  最后波峰: 帧索引 {max(frame_indices)}")

        if all_red_peaks:
            frame_indices = [p[0] for p in all_red_peaks]
            print(f"\n红色波峰分布:")
            print(f"  首个波峰: 帧索引 {min(frame_indices)}")
            print(f"  最后波峰: 帧索引 {max(frame_indices)}")

        # 显示前10个有波峰的帧
        print(f"\n前10个有波峰的帧:")
        for info in peaks_by_frame[:10]:
            print(f"  帧 {info['frame_index']}: 绿色={info['green_count']}, 红色={info['red_count']}")

        if len(peaks_by_frame) > 10:
            print(f"  ... 还有 {len(peaks_by_frame) - 10} 个帧")

        print("="*80 + "\n")

        return peaks_by_frame

    def analyze_intersections(self):
        """分析绿线交点分布"""
        print("\n" + "="*80)
        print("绿线交点分布分析")
        print("="*80)

        x_coords = []
        y_coords = []
        invalid_frames = []

        for frame in self.frames:
            frame_idx = frame.get('frame_index', -1)
            if 'intersection' in frame and isinstance(frame['intersection'], dict):
                used = frame['intersection'].get('used')
                if used and isinstance(used, list) and len(used) >= 2:
                    x_coords.append((frame_idx, used[0]))
                    y_coords.append((frame_idx, used[1]))
                else:
                    invalid_frames.append(frame_idx)
            else:
                invalid_frames.append(frame_idx)

        if not x_coords:
            print("\n未检测到有效交点")
            return

        print(f"\n有效交点数量: {len(x_coords)}")
        print(f"无效交点数量: {len(invalid_frames)}")

        x_values = [x[1] for x in x_coords]
        y_values = [y[1] for y in y_coords]

        print(f"\nX坐标统计:")
        print(f"  均值: {np.mean(x_values):.2f}")
        print(f"  标准差: {np.std(x_values):.2f}")
        print(f"  范围: [{np.min(x_values):.2f}, {np.max(x_values):.2f}]")
        if len(x_values) > 1:
            print(f"  最大抖动: {np.max(np.abs(np.diff(x_values))):.2f}")
            print(f"  平均抖动: {np.mean(np.abs(np.diff(x_values))):.2f}")

        print(f"\nY坐标统计:")
        print(f"  均值: {np.mean(y_values):.2f}")
        print(f"  标准差: {np.std(y_values):.2f}")
        print(f"  范围: [{np.min(y_values):.2f}, {np.max(y_values):.2f}]")
        if len(y_values) > 1:
            print(f"  最大抖动: {np.max(np.abs(np.diff(y_values))):.2f}")
            print(f"  平均抖动: {np.mean(np.abs(np.diff(y_values))):.2f}")

        if invalid_frames:
            print(f"\n无效交点帧索引: {invalid_frames[:20]}")
            if len(invalid_frames) > 20:
                print(f"  ... 还有 {len(invalid_frames) - 20} 个")

        print("="*80 + "\n")

    def plot_waveform(self, output_file: Optional[str] = None, frame_range: Optional[Tuple[int, int]] = None):
        """绘制波形图"""
        if not self.frames:
            print("没有数据可绘制")
            return

        # 过滤帧范围
        frames = self.frames
        if frame_range:
            start, end = frame_range
            frames = [f for f in self.frames if start <= f.get('frame_index', -1) <= end]

        if not frames:
            print(f"指定帧范围 {frame_range} 内没有数据")
            return

        frame_indices = [f.get('frame_index', i) for i, f in enumerate(frames)]
        roi2_values = [f.get('roi2_gray', np.nan) for f in frames]
        roi3_values = []
        thresholds = []

        # ROI3 灰度值和阈值
        for f in frames:
            if 'roi3' in f and isinstance(f['roi3'], dict):
                roi3_values.append(f['roi3'].get('gray', np.nan))
            else:
                roi3_values.append(np.nan)

            if 'threshold' in f and isinstance(f['threshold'], dict):
                thresholds.append(f['threshold'].get('used', np.nan))
            else:
                thresholds.append(np.nan)

        # 收集波峰信息
        peak_frames = []
        peak_colors = []
        for i, f in enumerate(frames):
            if 'peaks' in f and isinstance(f['peaks'], dict):
                peaks = f['peaks']
                green_count = len(peaks.get('green', []))
                red_count = len(peaks.get('red', []))
                if green_count > 0:
                    peak_frames.append((i, 'green'))
                if red_count > 0:
                    peak_frames.append((i, 'red'))

        # 创建图表
        fig = plt.figure(figsize=(16, 8))
        gs = GridSpec(2, 1, figure=fig, hspace=0.3)

        # ROI2 波形
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(frame_indices, roi2_values, 'g-', linewidth=1, label='ROI2 平均灰度', alpha=0.7)

        # 标记波峰
        for idx, color in peak_frames:
            frame_idx = frame_indices[idx]
            peak_color = 'red' if color == 'red' else 'green'
            ax1.plot(frame_idx, roi2_values[idx], 'o', color=peak_color,
                    markersize=10, markeredgecolor='black', markeredgewidth=1.5,
                    label=f'{color} peak' if idx == peak_frames[0][0] or
                          (idx == peak_frames[1][0] and len(peak_frames) > 1) else '')

        if not np.all(np.isnan(thresholds)):
            ax1.plot(frame_indices, thresholds, 'r--', linewidth=1, label='检测阈值', alpha=0.5)

        ax1.set_xlabel('帧索引')
        ax1.set_ylabel('灰度值')
        ax1.set_title('ROI2 波形图 (带波峰标记)')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right')

        # ROI3 波形 (如果有数据)
        ax2 = fig.add_subplot(gs[1, 0])
        if not np.all(np.isnan(roi3_values)):
            ax2.plot(frame_indices, roi3_values, 'm-', linewidth=1, label='ROI3 平均灰度', alpha=0.7)
            ax2.set_xlabel('帧索引')
            ax2.set_ylabel('灰度值')
            ax2.set_title('ROI3 波形图')
        else:
            ax2.text(0.5, 0.5, 'ROI3 数据不可用',
                    ha='center', va='center', transform=ax2.transAxes, fontsize=14)
            ax2.set_title('ROI3 波形图')

        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right')

        # 总标题
        session_id = self.meta.get('session_id', 'N/A') if self.meta else 'N/A'
        fig.suptitle(f'ROI 分析缓存 - 会话: {session_id}\n文件: {self.cache_file.name}',
                    fontsize=12, fontweight='bold')

        # 保存或显示
        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"\n波形图已保存到: {output_file}")
        else:
            plt.show()

        plt.close()

    def export_to_csv(self, output_file: Optional[str] = None):
        """导出到CSV文件"""
        import csv

        if output_file is None:
            output_file = self.cache_file.stem + '_export.csv'

        output_path = Path(output_file)

        if not self.frames:
            print("没有数据可导出")
            return

        # 收集所有可能的字段
        fieldnames = set()
        for frame in self.frames:
            fieldnames.update(frame.keys())

        # 排序字段名
        fieldnames = sorted(fieldnames)

        # 移除嵌套字段（如intersection, roi1_coords等）
        simple_fieldnames = [f for f in fieldnames if not any(
            f.startswith(prefix) for prefix in ['intersection', 'roi1_coords', 'roi2_coords', 'roi3_coords']
        )]

        # 添加展开的坐标字段
        extended_fieldnames = simple_fieldnames + [
            'intersection_x', 'intersection_y',
            'roi1_x1', 'roi1_y1', 'roi1_x2', 'roi1_y2',
            'roi2_x1', 'roi2_y1', 'roi2_x2', 'roi2_y2',
        ]

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=extended_fieldnames, extrasaction='ignore')
            writer.writeheader()

            for frame in self.frames:
                row = frame.copy()

                # 展开嵌套字段
                if 'intersection' in frame and frame['intersection']:
                    row['intersection_x'] = frame['intersection'].get('x')
                    row['intersection_y'] = frame['intersection'].get('y')

                if 'roi1_coords' in frame and frame['roi1_coords']:
                    coords = frame['roi1_coords']
                    if len(coords) == 4:
                        row['roi1_x1'], row['roi1_y1'], row['roi1_x2'], row['roi1_y2'] = coords

                if 'roi2_coords' in frame and frame['roi2_coords']:
                    coords = frame['roi2_coords']
                    if len(coords) == 4:
                        row['roi2_x1'], row['roi2_y1'], row['roi2_x2'], row['roi2_y2'] = coords

                writer.writerow(row)

        print(f"\n数据已导出到: {output_path}")
        print(f"总计 {len(self.frames)} 行记录")


def parse_frame_range(range_str: str) -> Tuple[int, int]:
    """解析帧范围字符串"""
    try:
        start, end = range_str.split('-')
        return int(start), int(end)
    except ValueError:
        raise argparse.ArgumentTypeError(f"无效的帧范围格式: {range_str}。期望格式: X-Y (例如: 100-200)")


def main():
    parser = argparse.ArgumentParser(
        description='SimpleFEM ROI分析缓存分析工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('cache_file', help='ROI分析缓存文件路径 (JSONL格式)')
    parser.add_argument('--summary', action='store_true', help='显示基本统计摘要')
    parser.add_argument('--peaks', action='store_true', help='详细分析波峰检测')
    parser.add_argument('--intersection', action='store_true', help='分析绿线交点分布')
    parser.add_argument('--waveform', action='store_true', help='绘制波形图')
    parser.add_argument('--export-csv', metavar='OUTPUT', const=None, nargs='?',
                       help='导出到CSV文件 (可选: 指定输出文件名)')
    parser.add_argument('--filter-frames', type=parse_frame_range,
                       help='只分析指定帧范围 (例如: 100-200)')
    parser.add_argument('--output-waveform', metavar='FILE',
                       help='保存波形图到指定文件而不是显示')

    args = parser.parse_args()

    # 如果没有指定任何分析选项，默认显示摘要
    if not any([args.summary, args.peaks, args.intersection, args.waveform, args.export_csv is not None]):
        args.summary = True

    try:
        analyzer = CacheAnalyzer(args.cache_file)

        if args.summary:
            analyzer.print_summary()

        if args.peaks:
            analyzer.analyze_peaks()

        if args.intersection:
            analyzer.analyze_intersections()

        if args.waveform:
            analyzer.plot_waveform(
                output_file=args.output_waveform,
                frame_range=args.filter_frames
            )

        if args.export_csv is not None:
            analyzer.export_to_csv(args.export_csv)

    except FileNotFoundError as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
