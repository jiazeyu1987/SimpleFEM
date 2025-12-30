"""
测试ROI3配置
"""
import json

with open("simple_fem_config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

roi3_config = config.get("roi_capture", {}).get("roi3_config", {})
roi3_extension_params = roi3_config.get("extension_params", {})

print("roi3_extension_params:", roi3_extension_params)
print("类型:", type(roi3_extension_params))
print("布尔值(bool):", bool(roi3_extension_params))

# 模拟代码中的条件判断
if roi3_extension_params:
    print("\n条件为True - ROI3代码应该执行")
else:
    print("\n条件为False - ROI3代码不会执行")

# 检查是否有实际参数
has_params = any(k in roi3_extension_params for k in ['left', 'right', 'top', 'bottom'])
print("\n是否有实际参数:", has_params)

# 检查参数值
for key in ['left', 'right', 'top', 'bottom']:
    value = roi3_extension_params.get(key)
    print(f"  {key}: {value} (type: {type(value).__name__})")
