"""
检查ROI1在帧335-344的数据
"""
import json

cache_file = "export/roi_analysis_cache_3(5次发射，1无效）_20251224_215925_f4648ff7f7c9.jsonl"

print("=== 检查帧335-344的ROI1数据 ===\n")

with open(cache_file, "r", encoding="utf-8") as f:
    for line in f:
        try:
            data = json.loads(line)
            frame_idx = data.get("frame_index", -1)

            if 335 <= frame_idx <= 344:
                roi1_gray = data.get("roi1_gray")
                roi2_gray = data.get("roi2_gray")
                threshold = data.get("threshold", {})

                print(f"帧{frame_idx}:")
                print(f"  ROI1灰度: {roi1_gray if roi1_gray else 'N/A'}")
                print(f"  ROI2灰度: {roi2_gray:.2f}")
                print(f"  阈值(ROI2): {threshold.get('used'):.2f}")
                print()

                if frame_idx == 342:
                    # 打印完整数据
                    print("=== 帧342完整数据 ===")
                    for key, value in data.items():
                        if key not in ['config', 'roi1', 'roi2_region', 'intersection', 'screen_size', 'ts_wall', 'ts_local']:
                            print(f"  {key}: {value}")
                    print()
        except Exception as e:
            print(f"Error: {e}")
