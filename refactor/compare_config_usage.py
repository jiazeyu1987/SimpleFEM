"""
配置使用对比工具

对比原始代码和重构代码如何使用配置文件
"""

import json
import os
import sys


def load_config():
    """加载配置文件"""
    config_path = "simple_fem_config.json"
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        return None

    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_config(config):
    """分析配置对检测模式的影响"""

    print("=" * 80)
    print("配置分析报告")
    print("=" * 80)

    # 提取关键配置
    hybrid_detection = config.get('hybrid_detection', {})
    roi1_detection = config.get('roi1_peak_detection', {})
    peak_detection = config.get('peak_detection', {})

    hybrid_enabled = hybrid_detection.get('enabled', False)
    roi1_enabled = roi1_detection.get('enabled', False)

    print("\n1. 关键配置项:")
    print(f"   hybrid_detection.enabled: {hybrid_enabled}")
    print(f"   roi1_peak_detection.enabled: {roi1_enabled}")

    print("\n2. 检测模式判断:")
    print(f"   hybrid_enabled AND roi1_enabled = {hybrid_enabled and roi1_enabled}")

    print("\n3. 执行流程分析:")

    if hybrid_enabled and roi1_enabled:
        print("   ✓ 将进入混合检测模式")
        print("   ├─ 情况1: ROI1缓冲区 > 0 → 执行混合检测")
        print("   ├─ 情况2: ROI1缓冲区 == 0 → 跳过检测（等待数据）")
        print("   └─ 不会进入ROI2独立检测模式")

        print("\n4. ⚠️  潜在问题:")
        print("   在视频开始阶段，ROI1缓冲区为空，会跳过所有波峰检测")
        print("   这可能导致 peak_statistics 文件中没有数据")

        print("\n5. 可能的解决方案:")
        print("   方案A: 关闭混合检测")
        print("     → 修改配置: hybrid_detection.enabled = false")
        print("     → 效果: 使用传统的ROI2独立检测")
        print("     → 优点: 立即开始检测，不会跳过任何帧")
        print("     → 缺点: 失去混合检测的优势")

        print("\n   方案B: 等待ROI1缓冲区积累数据")
        print("     → 保持当前配置")
        print("     → 效果: 等待100帧后自动使用混合检测")
        print("     → 优点: 一旦缓冲区满，使用混合检测")
        print("     → 缺点: 视频开始阶段的波峰会丢失")

        print("\n   方案C: 修改代码添加回退机制")
        print("     → 修改 refactor/orchestrator.py")
        print("     → 在ROI1数据不足时回退到ROI2检测")
        print("     → 优点: 不丢失任何波峰")
        print("     → 缺点: 与原始代码逻辑不一致")

    elif not hybrid_enabled and not roi1_enabled:
        print("   ✓ 将进入ROI2独立检测模式")
        print("   → 这是传统的波峰检测方式")
        print("   → 所有帧都会进行检测")

    else:
        print("   ✓ 将进入ROI2独立检测模式")
        print("   → 因为混合检测未完全启用")

    print("\n" + "=" * 80)
    print("配置详情")
    print("=" * 80)

    print("\n混合检测配置:")
    print(json.dumps(hybrid_detection, indent=2, ensure_ascii=False))

    print("\nROI1检测配置:")
    print(json.dumps(roi1_detection, indent=2, ensure_ascii=False))

    print("\nROI2检测配置:")
    print(json.dumps(peak_detection, indent=2, ensure_ascii=False))

    return {
        'hybrid_enabled': hybrid_enabled,
        'roi1_enabled': roi1_enabled,
        'will_skip_detection': hybrid_enabled and roi1_enabled
    }


def compare_with_export2():
    """对比export2的配置（如果存在）"""

    print("\n" + "=" * 80)
    print("检查export2目录")
    print("=" * 80)

    export2_dir = "export2"
    if not os.path.exists(export2_dir):
        print("❌ export2目录不存在")
        return

    # 查找CSV文件
    csv_files = [f for f in os.listdir(export2_dir) if f.endswith('.csv')]
    if not csv_files:
        print("❌ export2目录中没有CSV文件")
        return

    print(f"✓ 找到 {len(csv_files)} 个CSV文件")

    # 检查是否有波峰数据
    for csv_file in csv_files:
        csv_path = os.path.join(export2_dir, csv_file)
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # 跳过标题行
            data_lines = lines[1:] if len(lines) > 1 else []

            print(f"\n文件: {csv_file}")
            print(f"  总行数: {len(lines)}")
            print(f"  数据行数: {len(data_lines)}")

            if len(data_lines) > 0:
                print(f"  ✓ 有波峰数据")

                # 分析第一行数据
                first_data = data_lines[0].strip().split(',')
                if len(first_data) > 0:
                    print(f"  第一条数据帧索引: {first_data[1] if len(first_data) > 1 else 'N/A'}")
            else:
                print(f"  ❌ 没有波峰数据")

        except Exception as e:
            print(f"  ❌ 读取失败: {e}")


def suggest_next_step(analysis):
    """建议下一步操作"""

    print("\n" + "=" * 80)
    print("建议的下一步操作")
    print("=" * 80)

    if analysis.get('will_skip_detection'):
        print("\n由于当前配置会导致跳过波峰检测，建议：")
        print("\n1. 快速验证（推荐）:")
        print("   临时关闭混合检测，重新运行看是否有波峰")
        print("   → 修改 simple_fem_config.json:")
        print("     hybrid_detection.enabled = false")
        print("   → 运行: python -m refactor.main")
        print("   → 检查 export/peak_statistics_*.csv")

        print("\n2. 保持混合检测:")
        print("   如果确实需要使用混合检测，请确保视频足够长")
        print("   → ROI1缓冲区需要100帧才能开始检测")
        print("   → 视频开始阶段的波峰会丢失")

        print("\n3. 检查export2的配置:")
        print("   如果export2有数据而export没有，可能配置不同")
        print("   → 对比两个目录使用的配置文件")
        print("   → 或者查看export2使用的是哪个版本的代码")

    else:
        print("\n当前配置应该能正常检测波峰")
        print("如果还是没有数据，可能需要检查：")
        print("1. 阈值设置是否过高")
        print("2. 视频数据是否有效")
        print("3. 日志文件中的错误信息")


def main():
    """主函数"""
    print("SimpleFEM 配置分析工具")

    # 切换到项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)
    print(f"工作目录: {os.getcwd()}")

    # 加载配置
    config = load_config()
    if not config:
        return

    # 分析配置
    analysis = analyze_config(config)

    # 对比export2
    compare_with_export2()

    # 建议下一步
    suggest_next_step(analysis)


if __name__ == '__main__':
    main()
