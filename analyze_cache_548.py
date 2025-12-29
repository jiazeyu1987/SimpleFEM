"""
分析缓存文件中帧548的波峰检测情况
"""
import json
import os

cache_file = 'export/roi_analysis_cache_2（5次发射，1无效)_20251224_111509_9b1fd54fc752.jsonl'

# 读取缓存文件
records = []
with open(cache_file, 'r', encoding='utf-8') as f:
    for line_num, line in enumerate(f, 1):
        try:
            record = json.loads(line)
            frame_index = record.get('frame_index', -1)
            records.append({
                'line_num': line_num,
                'frame_index': frame_index,
                'record': record
            })
        except:
            pass

# 查找帧548附近的记录
print(f"=== 分析缓存文件: {os.path.basename(cache_file)} ===")
print(f"总记录数: {len(records)}")
print()

# 查找帧540-560范围内的所有记录
target_records = [r for r in records if 540 <= r['frame_index'] <= 560]
print(f"=== 帧540-560范围内的记录 ({len(target_records)}条) ===")

for r in target_records:
    rec = r['record']
    frame_idx = rec['frame_index']

    # 检查是否有波峰检测结果
    has_peaks = 'peak_detection' in rec
    peak_info = rec.get('peak_detection', {})

    roi2_peaks = peak_info.get('roi2_peaks', {})
    green_peaks = roi2_peaks.get('green_peaks', [])
    red_peaks = roi2_peaks.get('red_peaks', [])

    # 检查是否有波峰在548附近
    peak_548_found = False
    for peak in green_peaks + red_peaks:
        peak_frame = peak.get('peak_frame', -1)
        peak_start = peak.get('start_frame', -1)
        peak_end = peak.get('end_frame', -1)
        peak_max = peak.get('peak_max_value', 0)

        # 检查波峰是否包含548帧
        if peak_start <= 548 <= peak_end:
            peak_548_found = True
            print(f"\n行{r['line_num']}: 帧{frame_idx}")
            print(f"  发现包含帧548的波峰!")
            print(f"  波峰类型: {'green' if peak in green_peaks else 'red'}")
            print(f"  波峰范围: [{peak_start}, {peak_end}]")
            print(f"  波峰最大值: {peak_max:.2f} @ 帧{peak_frame}")
            print(f"  前置平均: {peak.get('pre_peak_avg', 0):.2f}")
            print(f"  后置平均: {peak.get('post_peak_avg', 0):.2f}")
            print(f"  帧差: {peak.get('frame_diff', 0):.3f}")

    if not peak_548_found and has_peaks and (green_peaks or red_peaks):
        print(f"行{r['line_num']}: 帧{frame_idx} - 检测到{len(green_peaks)}绿+{len(red_peaks)}红波峰，但都不包含帧548")

print()
print("=== 检查548附近的记录中是否有波峰被标记 ===")

# 查找frame_index=548的记录
frame_548_records = [r for r in records if r['frame_index'] == 548]
if frame_548_records:
    print(f"\n找到 {len(frame_548_records)} 条帧548的记录:")
    for r in frame_548_records[:3]:  # 只显示前3条
        rec = r['record']
        print(f"  行{r['line_num']}: {json.dumps(rec, ensure_ascii=False)[:200]}...")
else:
    print("没有找到frame_index=548的记录")

print()
print("=== 检查波峰548的峰值(117.17)是否出现在其他帧 ===")

# 查找包含峰值117.17附近的波峰
target_max = 117.17
for r in records:
    rec = r['record']
    peak_info = rec.get('peak_detection', {})
    roi2_peaks = peak_info.get('roi2_peaks', {})
    green_peaks = roi2_peaks.get('green_peaks', [])
    red_peaks = roi2_peaks.get('red_peaks', [])

    for peak in green_peaks + red_peaks:
        peak_max = peak.get('peak_max_value', 0)
        if abs(peak_max - target_max) < 0.5:
            peak_frame = peak.get('peak_frame', -1)
            peak_start = peak.get('start_frame', -1)
            peak_end = peak.get('end_frame', -1)
            print(f"行{r['line_num']}: 帧{rec['frame_index']} - 峰值{peak_max:.2f} @ 帧{peak_frame}, 范围[{peak_start},{peak_end}]")
