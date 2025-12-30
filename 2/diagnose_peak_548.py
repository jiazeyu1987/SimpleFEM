"""
诊断波峰548未被检测的原因
"""
import json
import csv

# 读取配置
with open('simple_fem_config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

# 读取CSV文件
csv_file = 'export2/peak_statistics_2（5次发射，1无效)_20251224_005115.csv'

peaks = []
with open(csv_file, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        frame_index = int(row['frame_index'])
        peaks.append({
            'frame_index': frame_index,
            'peak_type': row['peak_type'],
            'pre_avg': float(row['pre_peak_avg']),
            'post_avg': float(row['post_peak_avg']),
            'frame_diff': float(row['frame_diff']),
            'peak_max': float(row['peak_max_value']),
            'pre_frame_start': int(row['pre_peak_frame_start']),
            'pre_frame_end': int(row['pre_peak_frame_end']),
            'post_frame_start': int(row['post_peak_frame_start']),
            'post_frame_end': int(row['post_peak_frame_end'])
        })

# 查找548附近的波峰
print("=== 配置信息 ===")
peak_conf = config['peak_detection']
print(f"pre_frame_offset: {peak_conf.get('pre_frame_offset', 0)}")
print(f"post_frame_offset: {peak_conf.get('post_frame_offset', 0)}")
print(f"difference_threshold: {peak_conf.get('difference_threshold', 1.5)}")
print()

print("=== 检测到的波峰 ===")
for p in peaks:
    print(f"帧{p['frame_index']}: {p['peak_type']}, diff={p['frame_diff']:.3f}, max={p['peak_max']:.2f}, pre=[{p['pre_frame_start']},{p['pre_frame_end']}], post=[{p['post_frame_start']},{p['post_frame_end']}]")
print()

# 检查548是否在波峰附近
print("=== 分析548附近 ===")
target_frame = 548
nearby_peaks = [p for p in peaks if abs(p['frame_index'] - target_frame) <= 100]

if nearby_peaks:
    print(f"548附近100帧内检测到{len(nearby_peaks)}个波峰:")
    for p in nearby_peaks:
        distance = abs(p['frame_index'] - target_frame)
        print(f"  - 帧{p['frame_index']} (距离{distance}帧): {p['peak_type']}, diff={p['frame_diff']:.3f}, max={p['peak_max']:.2f}")
else:
    print("548附近100帧内没有检测到任何波峰")
print()

# 分析可能的548波峰参数
print("=== 推测548波峰参数 ===")
# 假设548是一个波峰，且 pre_frame_offset=-6, post_frame_offset=3
pre_offset = peak_conf.get('pre_frame_offset', 0)
post_offset = peak_conf.get('post_frame_offset', 0)
pre_frames = peak_conf.get('pre_post_avg_frames', 5)

# 如果548是波峰结束帧，推算前置和后置帧范围
# 假设波峰宽度约10帧
peak_start = 548 - 5  # 假设波峰从543开始
peak_end = 548

pre_frame_start = peak_start - pre_frames + pre_offset
pre_frame_end = peak_start - 1 + pre_offset
post_frame_start = peak_end + 1 + post_offset
post_frame_end = peak_end + pre_frames + post_offset

print(f"如果548是波峰结束帧:")
print(f"  推测前置帧: [{pre_frame_start}, {pre_frame_end}]")
print(f"  推测后置帧: [{post_frame_start}, {post_frame_end}]")
print()

# 检查去重配置
dup_config = config.get('deduplication', {})
print("=== 去重配置 ===")
print(f"consecutive_frame_window: {dup_config.get('consecutive_frame_window', 40)}")
print(f"consecutive_deduplication_enabled: {dup_config.get('consecutive_deduplication_enabled', True)}")
print(f"cross_color_deduplication_enabled: {dup_config.get('cross_color_deduplication_enabled', True)}")
print()

# 检查是否有波峰与548的峰值相同
print("=== 峰值去重检查 ===")
peak_548_max = 117.17  # 用户提供的峰值
print(f"548的峰值: {peak_548_max}")

for p in peaks:
    if abs(p['peak_max'] - peak_548_max) < 0.5:  # 峰值相近
        distance = abs(p['frame_index'] - target_frame)
        print(f"  - 帧{p['frame_index']} (距离{distance}帧): 峰值{p['peak_max']:.2f}, 类型{p['peak_type']}")
