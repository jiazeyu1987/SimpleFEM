"""
检查帧93的详细数据
"""
import json
import glob

cache_files = glob.glob("export/roi_analysis_cache_*2335*.jsonl")
if cache_files:
    cache_file = cache_files[0]
    print(f"检查缓存: {cache_file}\n")

    with open(cache_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                frame_idx = data.get("frame_index", -1)

                # 找到帧93的数据
                if frame_idx == 93:
                    print("=== 帧93的完整数据 ===")
                    for key, value in data.items():
                        if key not in ['config', 'roi1', 'roi2_region', 'intersection', 'screen_size', 'ts_wall', 'ts_local', 'host']:
                            print(f"{key}: {value}")

                    # 特别检查ROI3
                    if 'roi3' in str(data):
                        print("\n=== ROI3相关 ===")
                        roi3_gray = data.get("roi3_gray")
                        print(f"roi3_gray: {roi3_gray}")

                    break
            except:
                pass
else:
    print("找不到缓存文件")
