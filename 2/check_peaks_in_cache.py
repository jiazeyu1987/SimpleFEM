"""
检查缓存中的波峰记录
"""
import json
import glob

cache_files = glob.glob("export/roi_analysis_cache_*.jsonl")
if cache_files:
    cache_files.sort(key=lambda x: x.split('_')[-1])  # 按时间戳排序
    latest_cache = cache_files[-1]
    print(f"检查缓存文件: {latest_cache}\n")

    found_peaks = []
    with open(latest_cache, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                frame_idx = data.get("frame_index", -1)

                peaks = data.get("peaks", {})
                green = peaks.get("green", [])
                red = peaks.get("red", [])

                if green or red:
                    found_peaks.append({
                        'frame': frame_idx,
                        'green': green,
                        'red': red
                    })

            except:
                pass

    print(f"总共找到 {len(found_peaks)} 个有波峰的帧")
    print("\n波峰列表：")
    for p in found_peaks:
        print(f"  帧{p['frame']}: 绿色={len(p['green'])}个, 红色={len(p['red'])}个")
        for g in p['green']:
            print(f"    GREEN: abs=[{g.get('abs_start')},{g.get('abs_end')}]")
        for r in p['red']:
            print(f"    RED: abs=[{r.get('abs_start')},{r.get('abs_end')}]")
else:
    print("找不到缓存文件")
