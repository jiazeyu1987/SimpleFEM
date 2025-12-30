"""
分析ROI1检测到的波峰和实际统计的波峰数量差异
"""
import json

cache_file = 'export/roi_analysis_cache_3(5次发射，1无效）_20251224_135910_cfa55121cadb.jsonl'

print('=== 所有被处理的波峰（包括被过滤的）===')

all_processed = []
added_peaks = []
skipped_peaks = []

with open(cache_file, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            data = json.loads(line)
            stats = data.get('stats_write', [])

            for stat in stats:
                if isinstance(stat, dict):
                    frame_idx = stat.get('frame_index', -1)
                    action = stat.get('action', '')
                    skip_reason = stat.get('skip_reason', '')
                    peak_max = stat.get('peak_max_value', 0)
                    pre_avg = stat.get('pre_peak_avg', 0)
                    post_avg = stat.get('post_peak_avg', 0)

                    peak_info = {
                        'frame_idx': frame_idx,
                        'action': action,
                        'skip_reason': skip_reason,
                        'peak_max': peak_max,
                        'pre_avg': pre_avg,
                        'post_avg': post_avg
                    }

                    all_processed.append(peak_info)

                    if action == 'added':
                        added_peaks.append(peak_info)
                        status = f'✓ 添加'
                    else:
                        skipped_peaks.append(peak_info)
                        status = f'✗ 过滤 ({skip_reason})'

                    print(f'帧{frame_idx}: max={peak_max:.2f}, pre={pre_avg:.2f}, post={post_avg:.2f}, {status}')
        except:
            pass

print(f'\n=== 统计汇总 ===')
print(f'总计处理: {len(all_processed)} 个波峰')
print(f'成功添加: {len(added_peaks)} 个波峰')
print(f'被过滤: {len(skipped_peaks)} 个波峰')

print(f'\n=== 被过滤的波峰详情 ===')
for p in skipped_peaks:
    print(f"  帧{p['frame_idx']}: max={p['peak_max']:.2f}, 原因={p['skip_reason']}")

print(f'\n=== 成功添加的波峰 ===')
for p in added_peaks:
    print(f"  帧{p['frame_idx']}: max={p['peak_max']:.2f}")
