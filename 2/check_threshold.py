"""
检查实际运行时的数据
"""
import json
import glob
import os

cache_files = glob.glob("export/roi_analysis_cache_*.jsonl")
if cache_files:
    cache_files.sort(key=os.path.getmtime)
    latest_cache = cache_files[-1]
    print(f"检查缓存文件: {latest_cache}\n")

    frame_count = 0
    with open(latest_cache, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                frame_idx = data.get("frame_index", -1)

                if frame_count < 5:
                    print(f"=== 帧{frame_idx} ===")

                    # 打印所有关键字段
                    for key in data.keys():
                        if key not in ['buffer', 'peaks', 'detection', 'stats_write']:
                            print(f"  {key}: {data[key]}")

                    # 检查 buffer
                    buffer = data.get("buffer", {})
                    print(f"  buffer keys: {list(buffer.keys())}")

                    # 尝试获取 gray_buffer
                    if "gray" in buffer:
                        gray_buffer = buffer["gray"]
                        print(f"  gray_buffer 长度: {len(gray_buffer)}")
                        if gray_buffer:
                            print(f"  gray_buffer 最近5帧: {gray_buffer[-5:]}")
                            print(f"  gray_buffer 最大值: {max(gray_buffer):.2f}")
                    else:
                        print("  ⚠️ buffer 中没有 'gray' 字段!")

                    # 检查 detection
                    detection = data.get("detection", {})
                    if detection:
                        print(f"  detection: {detection}")

                    print()
                    frame_count += 1
            except Exception as e:
                print(f"Error parsing frame: {e}")
                pass
else:
    print("找不到缓存文件")
