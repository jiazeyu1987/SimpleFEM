"""
检查330~350帧之间的波峰检测情况
"""
import json

cache_file = "export/roi_analysis_cache_3(5次发射，1无效）_20251224_215925_f4648ff7f7c9.jsonl"

print("=== 帧330-350的详细数据 ===\n")

with open(cache_file, "r", encoding="utf-8") as f:
    for line in f:
        try:
            data = json.loads(line)
            frame_idx = data.get("frame_index", -1)

            if 330 <= frame_idx <= 350:
                print(f"=== 帧{frame_idx} ===")

                # ROI2灰度值
                roi2_gray = data.get("roi2_gray", 0)
                print(f"  roi2_gray: {roi2_gray:.2f}")

                # 阈值信息
                threshold = data.get("threshold", {})
                print(f"  threshold.used: {threshold.get('used')}")
                print(f"  protection_active: {threshold.get('protection_active')}")

                # 波峰
                peaks = data.get("peaks", {})
                green = peaks.get("green", [])
                red = peaks.get("red", [])
                green_raw = peaks.get("green_raw", [])
                red_raw = peaks.get("red_raw", [])

                if green_raw or red_raw:
                    print(f"  green_raw: {green_raw}")
                    print(f"  red_raw: {red_raw}")

                if green or red:
                    print(f"  green: {green}")
                    print(f"  red: {red}")

                # stats_write
                stats = data.get("stats_write", [])
                if stats:
                    print(f"  stats_write: {len(stats)}条")
                    for stat in stats:
                        if isinstance(stat, dict):
                            print(f"    frame={stat.get('frame_index')}, action={stat.get('action')}, skip_reason={stat.get('skip_reason')}")

                print()
        except Exception as e:
            print(f"Error: {e}")

print("\n=== 分析 ===")
print("检查：")
print("1. ROI2灰度值是否超过阈值？")
print("2. 是否有raw波峰但被过滤了？")
print("3. 是否被stats_write跳过了？")
