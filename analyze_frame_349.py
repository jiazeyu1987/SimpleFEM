"""
详细检查帧349附近的所有数据
"""
import json

cache_file = "export/roi_analysis_cache_3(5次发射，1无效）_20251224_150439_467809771299.jsonl"

print("=== 帧340-360的详细数据（包括buffer和threshold）===")
with open(cache_file, "r", encoding="utf-8") as f:
    for line in f:
        try:
            data = json.loads(line)
            frame_idx = data.get("frame_index", -1)

            if 340 <= frame_idx <= 360:
                print(f"\n=== 帧{frame_idx} ===")

                # 检查peaks
                peaks = data.get("peaks", {})
                green = peaks.get("green", [])
                red = peaks.get("red", [])
                green_raw = peaks.get("green_raw", [])
                red_raw = peaks.get("red_raw", [])

                print(f"peaks.green: {green}")
                print(f"peaks.red: {red}")
                print(f"peaks.green_raw: {green_raw}")
                print(f"peaks.red_raw: {red_raw}")

                # 检查buffer
                buffer = data.get("buffer", {})
                print(f"buffer: {buffer}")

                # 检查threshold
                threshold = data.get("threshold", {})
                print(f"threshold.used: {threshold.get('used')}")

                # 检查stats_write
                stats = data.get("stats_write", [])
                print(f"stats_write count: {len(stats)}")
                for stat in stats:
                    if isinstance(stat, dict):
                        print(f"  frame={stat.get('frame_index')}, action={stat.get('action')}")
        except Exception as e:
            print(f"Error: {e}")

print("\n=== 分析：帧349为什么没被检测到 ===")
print("可能原因：")
print("1. ROI1平均灰度值 < threshold (63.0)")
print("2. ROI1平均灰度值 < 自适应阈值")
print("3. 波峰宽度 < min_peak_width (5帧)")
print("4. 前后安静区 < silence_frames (5帧)")
print("5. 被margin_frames过滤（与相邻波峰太近）")
