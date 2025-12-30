"""
最终诊断：验证所有参数
"""
import json
import sys

print("=" * 80)
print("数据流诊断报告")
print("=" * 80)

# 1. 配置文件检查
print("\n【配置文件】")
with open("simple_fem_config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

peak_conf = config.get("peak_detection", {})
roi3_override = peak_conf.get("roi3_override", {})
print(f"  roi3_override.enabled: {roi3_override.get('enabled')}")
print(f"  roi3_override.threshold: {roi3_override.get('threshold')}")

roi3_config = config.get("roi_capture", {}).get("roi3_config", {})
roi3_params = roi3_config.get("extension_params", {})
print(f"  roi3_config.extension_params: {roi3_params}")

# 2. 代码检查
print("\n【代码检查】")
with open("safe_peak_statistics.py", "r", encoding="utf-8") as f:
    content = f.read()

has_green_to_red = "roi3_peak_max_frame < 110" in content and "final_peak_type = \"red\"" in content
print(f"  GREEN->RED逻辑存在: {has_green_to_red}")

has_debug_log = "[DEBUG ROI3]" in content
print(f"  调试日志已添加: {has_debug_log}")

# 3. CSV数据检查
print("\n【CSV数据】")
import csv
import glob
import os

csv_files = glob.glob("export/peak_statistics_*.csv")
if csv_files:
    csv_files.sort(key=os.path.getmtime)
    latest_csv = csv_files[-1]
    print(f"  最新CSV: {os.path.basename(latest_csv)}")

    with open(latest_csv, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = row.get('frame_index', '')
            roi3_frame = row.get('roi3_peak_max_frame', '')
            peak_type = row.get('peak_type', '')

            if frame == '93' and roi3_frame and roi3_frame != '-1':
                roi3_frame_int = int(float(roi3_frame))
                print(f"  帧{frame}: {peak_type}, roi3_peak_max_frame={roi3_frame_int}")
                print(f"    应该触发覆盖: {peak_type}='green' and {roi3_frame_int} < 110")
                print(f"    实际触发: {peak_type}='red' (覆盖成功) or {peak_type}='green' (覆盖失败)")
                break

print("\n" + "=" * 80)
print("诊断结论")
print("=" * 80)
print("""
如果CSV中显示绿色且roi3_peak_max_frame<110，说明覆盖逻辑未执行。

可能原因：
1. roi3_curve为空 - 需要查看 [DEBUG ROI3] roi3_curve_len
2. roi3_override_enabled为False - 需要查看 [DEBUG ROI3] roi3_override_enabled
3. 代码未更新 - 需要重新启动 simple_roi_daemon.py

请运行程序并查看控制台输出中的 [DEBUG ROI3] 日志。
""")
