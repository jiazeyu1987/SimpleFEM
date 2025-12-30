"""
检查帧349和785附近的波峰检测情况
"""
import json
import glob

# 找最新的缓存文件
cache_files = glob.glob("export/roi_analysis_cache_3*150439*.jsonl")
if cache_files:
    cache_file = cache_files[0]
    print(f"检查缓存: {cache_file}\n")

    # 检查帧349和785附近的波峰
    print("=== 帧340-360范围内的波峰 ===")
    with open(cache_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                frame_idx = data.get("frame_index", -1)
                peaks = data.get("peaks", {})
                green = peaks.get("green", [])
                red = peaks.get("red", [])

                if 340 <= frame_idx <= 360:
                    if green or red:
                        print(f"帧{frame_idx}:")
                        for g in green:
                            abs_start = g.get("abs_start", -1)
                            abs_end = g.get("abs_end", -1)
                            print(f"  GREEN abs=[{abs_start},{abs_end}]")
                        for r in red:
                            abs_start = r.get("abs_start", -1)
                            abs_end = r.get("abs_end", -1)
                            print(f"  RED abs=[{abs_start},{abs_end}]")
            except:
                pass

    print("\n=== 帧775-795范围内的波峰 ===")
    with open(cache_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                frame_idx = data.get("frame_index", -1)
                peaks = data.get("peaks", {})
                green = peaks.get("green", [])
                red = peaks.get("red", [])

                if 775 <= frame_idx <= 795:
                    if green or red:
                        print(f"帧{frame_idx}:")
                        for g in green:
                            abs_start = g.get("abs_start", -1)
                            abs_end = g.get("abs_end", -1)
                            print(f"  GREEN abs=[{abs_start},{abs_end}]")
                        for r in red:
                            abs_start = r.get("abs_start", -1)
                            abs_end = r.get("abs_end", -1)
                            print(f"  RED abs=[{abs_start},{abs_end}]")
            except:
                pass

    # 检查stats_write
    print("\n=== 所有被处理的波峰（包括349和785附近）===")
    with open(cache_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                stats = data.get("stats_write", [])

                for stat in stats:
                    if isinstance(stat, dict):
                        frame = stat.get("frame_index", -1)
                        # 只显示340-360和775-795范围的
                        if (340 <= frame <= 360) or (775 <= frame <= 795):
                            action = stat.get("action", "")
                            skip_reason = stat.get("skip_reason", "")
                            max_val = stat.get("peak_max_value", 0)
                            print(f"帧{frame}: max={max_val:.2f}, action={action}, reason={skip_reason}")
            except:
                pass
else:
    print("找不到对应的缓存文件")
