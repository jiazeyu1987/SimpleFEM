"""
检查帧442的详细数据
"""
import json

cache_file = 'export/roi_analysis_cache_3(5次发射，1无效）_20251224_135910_cfa55121cadb.jsonl'

print('=== 帧442的详细数据 ===')

with open(cache_file, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            data = json.loads(line)
            frame_idx = data.get('frame_index', -1)
            if frame_idx == 442:
                print('Frame Index:', frame_idx)

                print('\n=== peaks字段 ===')
                peaks = data.get('peaks', {})
                for key, value in peaks.items():
                    print(f'  {key}: {value}')

                print('\n=== detection字段 ===')
                detection = data.get('detection', {})
                for key, value in detection.items():
                    print(f'  {key}: {value}')

                print('\n=== stats_write字段 ===')
                stats = data.get('stats_write', [])
                for stat in stats:
                    if isinstance(stat, dict):
                        print(f'  frame_index: {stat.get("frame_index")}')
                        print(f'  action: {stat.get("action")}')
                        print(f'  peak_max_value: {stat.get("peak_max_value")}')
                        print(f'  detection_method: {stat.get("detection_method")}')
                        print(f'  detection_strategy: {stat.get("detection_strategy")}')
                        print(f'  roi1_peak_id: {stat.get("roi1_peak_id")}')
                        print(f'  roi1_peak_max: {stat.get("roi1_peak_max")}')

                print('\n=== 其他字段 ===')
                print(f'buffer: {data.get("buffer")}')
                print(f'threshold: {data.get("threshold")}')
        except Exception as e:
            print(f'Error: {e}')
            pass
