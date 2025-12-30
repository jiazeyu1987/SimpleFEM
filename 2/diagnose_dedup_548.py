"""
模拟去重逻辑，找出波峰548为什么被判定为重复
"""
import json

# 读取缓存文件，找出所有被检测到的波峰
cache_file = 'export/roi_analysis_cache_2（5次发射，1无效)_20251224_111509_9b1fd54fc752.jsonl'

all_peaks = []

with open(cache_file, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            data = json.loads(line)
            stats = data.get('stats_write', [])
            for stat in stats:
                if isinstance(stat, dict):
                    all_peaks.append({
                        'frame_index': stat.get('frame_index'),
                        'peak_type': stat.get('peak_type'),
                        'pre_avg': stat.get('pre_peak_avg'),
                        'post_avg': stat.get('post_peak_avg'),
                        'peak_max': stat.get('peak_max_value'),
                        'action': stat.get('action'),
                        'skip_reason': stat.get('skip_reason')
                    })
        except:
            pass

# 找出被接受的波峰（action != 'skipped'）
accepted_peaks = [p for p in all_peaks if p['action'] != 'skipped']

# 找出被跳过的波峰
skipped_peaks = [p for p in all_peaks if p['action'] == 'skipped']

print(f'=== 总波峰数: {len(all_peaks)} ===')
print(f'被接受: {len(accepted_peaks)}')
print(f'被跳过: {len(skipped_peaks)}')

print(f'\n=== 被接受的波峰 ===')
for p in accepted_peaks:
    print(f"  帧{p['frame_index']}: {p['peak_type']}, pre={p['pre_avg']:.2f}, post={p['post_avg']:.2f}, max={p['peak_max']:.2f}")

print(f'\n=== 被跳过的波峰（跳过原因）===')
for p in skipped_peaks:
    print(f"  帧{p['frame_index']}: {p['peak_type']}, skip_reason={p['skip_reason']}")

# 模拟去重逻辑，找出波峰548被哪个波峰判定为重复
print(f'\n=== 模拟去重逻辑检查波峰548 ===')
peak_548 = None
for p in all_peaks:
    if p['frame_index'] == 548:
        peak_548 = p
        break

if peak_548:
    print(f"波峰548: pre={peak_548['pre_avg']:.2f}, post={peak_548['post_avg']:.2f}, max={peak_548['peak_max']:.2f}")
    print(f"skip_reason: {peak_548['skip_reason']}")

    print(f"\n检查与前面波峰的相似度（容差2.0, 时间窗口200帧）:")
    for recent in accepted_peaks:
        if recent['frame_index'] < peak_548['frame_index']:
            frame_diff = abs(peak_548['frame_index'] - recent['frame_index'])
            if frame_diff <= 200:
                pre_diff = abs(peak_548['pre_avg'] - recent['pre_avg'])
                post_diff = abs(peak_548['post_avg'] - recent['post_avg'])

                match = pre_diff <= 2.0 and post_diff <= 2.0

                print(f"  vs 帧{recent['frame_index']}: 时间差={frame_diff}帧, pre_diff={pre_diff:.2f}, post_diff={post_diff:.2f} {'=> 匹配！(可能重复)' if match else ''}")
else:
    print("没有找到波峰548")
