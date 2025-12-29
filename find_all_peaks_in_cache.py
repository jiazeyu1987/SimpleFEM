"""
查找缓存中所有检测到的波峰（包括绿色和红色）
"""
import json

cache_file = 'export/roi_analysis_cache_3(5次发射，1无效）_20251224_135910_cfa55121cadb.jsonl'

print('=== 缓存中所有检测到的波峰（绿色+红色）===')

all_peaks = []

with open(cache_file, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            data = json.loads(line)
            frame_idx = data.get('frame_index', -1)
            peaks = data.get('peaks', {})

            # 绿色波峰
            green = peaks.get('green', [])
            for peak in green:
                abs_start = peak.get('abs_start', -1)
                abs_end = peak.get('abs_end', -1)
                all_peaks.append({
                    'frame_idx': frame_idx,
                    'color': 'green',
                    'abs_start': abs_start,
                    'abs_end': abs_end
                })
                print(f'帧{frame_idx}: GREEN abs=[{abs_start},{abs_end}]')

            # 红色波峰
            red = peaks.get('red', [])
            for peak in red:
                abs_start = peak.get('abs_start', -1)
                abs_end = peak.get('abs_end', -1)
                all_peaks.append({
                    'frame_idx': frame_idx,
                    'color': 'red',
                    'abs_start': abs_start,
                    'abs_end': abs_end
                })
                print(f'帧{frame_idx}: RED abs=[{abs_start},{abs_end}]')
        except:
            pass

print(f'\n=== 缓存中总计检测到 {len(all_peaks)} 个波峰 ===')

# 按绝对帧位置排序
all_peaks_sorted = sorted(all_peaks, key=lambda x: x['abs_start'])

print('\n=== 按绝对帧位置排序 ===')
for i, peak in enumerate(all_peaks_sorted):
    print(f"{i+1}. 帧{peak['frame_idx']}: {peak['color'].upper()}, abs=[{peak['abs_start']},{peak['abs_end']}]")
