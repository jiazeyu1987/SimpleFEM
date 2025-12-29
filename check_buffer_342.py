"""
检查帧342附近的缓冲区数据
"""
import json

cache_file = "export/roi_analysis_cache_3(5次发射，1无效）_20251224_215925_f4648ff7f7c9.jsonl"

print("=== 检查帧342的完整数据 ===\n")

with open(cache_file, "r", encoding="utf-8") as f:
    for line in f:
        try:
            data = json.loads(line)
            frame_idx = data.get("frame_index", -1)

            if frame_idx == 342:
                import pprint
                pprint.pprint(data, width=200)
                break
        except:
            pass
