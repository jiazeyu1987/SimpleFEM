"""
诊断ROI3覆盖功能
"""
import json

# 读取配置
with open("simple_fem_config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

print("=== 配置检查 ===")
peak_detection = config.get("peak_detection", {})
roi3_override = peak_detection.get("roi3_override", {})
print(f"ROI3 override enabled: {roi3_override.get('enabled')}")
print(f"ROI3 override threshold: {roi3_override.get('threshold')}")

print("\n=== 代码检查 ===")
import os
if os.path.exists("safe_peak_statistics.py"):
    with open("safe_peak_statistics.py", "r", encoding="utf-8") as f:
        content = f.read()
        if "roi3_peak_max_frame < 110" in content:
            print("[OK] 代码包含 'roi3_peak_max_frame < 110' 逻辑")
            # 统计出现次数
            count = content.count("roi3_peak_max_frame < 110")
            print(f"  出现次数: {count}次")
        else:
            print("[FAIL] 代码不包含 'roi3_peak_max_frame < 110' 逻辑")

        if "新增覆盖逻辑: GREEN -> RED" in content:
            print("[OK] 代码包含 '新增覆盖逻辑: GREEN -> RED' 注释")
        else:
            print("[FAIL] 代码不包含 '新增覆盖逻辑: GREEN -> RED' 注释")

print("\n=== 实际运行检查 ===")
# 检查最新的CSV文件
import glob
import csv

csv_files = glob.glob("export/peak_statistics_*.csv")
if csv_files:
    csv_files.sort(key=os.path.getmtime)
    latest_csv = csv_files[-1]
    print(f"最新CSV: {os.path.basename(latest_csv)}")

    # 读取CSV
    with open(latest_csv, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        green_early_count = 0
        for row in reader:
            peak_type = row.get('peak_type', '')
            roi3_frame = row.get('roi3_peak_max_frame', '')
            if roi3_frame and roi3_frame != '-1':
                try:
                    roi3_frame_int = int(float(roi3_frame))
                    if peak_type == 'green' and roi3_frame_int < 110:
                        green_early_count += 1
                        print(f"  帧{row['frame_index']}: {peak_type}, roi3_peak_max_frame={roi3_frame_int} < 110 ✓")
                except:
                    pass

    if green_early_count == 0:
        print("  没有找到 roi3_peak_max_frame < 110 的绿色波峰")
        print("  这是正常的，说明覆盖逻辑生效了（都已改为红色）")
else:
    print("找不到CSV文件")

print("\n=== 建议 ===")
print("1. 如果配置 enabled=false，请在 simple_fem_config.json 中设置为 true")
print("2. 重启 simple_roi_daemon.py")
print("3. 检查日志中是否有 '[DEBUG] ROI3 early-peak override applied'")
