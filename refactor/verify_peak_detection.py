"""
波峰检测问题快速验证脚本

用于诊断为什么重构代码生成的 peak_statistics 没有波峰数据
"""

import json
import os
import sys

def check_config():
    """检查配置文件"""
    print("=" * 60)
    print("步骤1: 检查配置文件")
    print("=" * 60)

    config_path = "simple_fem_config.json"
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        return None

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    hybrid_enabled = config.get('hybrid_detection', {}).get('enabled', False)
    roi1_enabled = config.get('roi1_peak_detection', {}).get('enabled', False)

    print(f"✓ 读取配置文件: {config_path}")
    print(f"  - hybrid_detection.enabled: {hybrid_enabled}")
    print(f"  - roi1_peak_detection.enabled: {roi1_enabled}")

    if hybrid_enabled and roi1_enabled:
        print("\n⚠️  检测到潜在问题:")
        print("  混合检测和ROI1检测都已启用")
        print("  在视频开始阶段，ROI1缓冲区为空，会跳过所有波峰检测")
        print("  这可能导致 peak_statistics 文件中没有数据")
        print("\n建议:")
        print("  1. 临时关闭混合检测验证问题")
        print("  2. 或者等待ROI1缓冲区积累足够数据（需要处理100+帧）")
        return {
            'hybrid_enabled': hybrid_enabled,
            'roi1_enabled': roi1_enabled,
            'problem_detected': True
        }
    else:
        print("\n✓ 配置看起来正常")
        return {
            'hybrid_enabled': hybrid_enabled,
            'roi1_enabled': roi1_enabled,
            'problem_detected': False
        }

def check_logs():
    """检查日志文件"""
    print("\n" + "=" * 60)
    print("步骤2: 检查日志文件")
    print("=" * 60)

    log_dir = "logs"
    if not os.path.exists(log_dir):
        print(f"❌ 日志目录不存在: {log_dir}")
        return

    log_files = [f for f in os.listdir(log_dir) if f.endswith('.log')]
    if not log_files:
        print(f"❌ 日志目录中没有日志文件")
        return

    # 读取最新的日志文件
    log_file = os.path.join(log_dir, sorted(log_files)[-1])
    print(f"✓ 读取最新日志文件: {log_file}")

    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # 查找关键日志
        roi1_insufficient_count = 0
        roi2_detection_count = 0
        hybrid_detection_count = 0

        for line in lines:
            if "ROI1数据不足" in line:
                roi1_insufficient_count += 1
            elif "ROI2独立检测" in line:
                roi2_detection_count += 1
            elif "混合检测模式" in line:
                hybrid_detection_count += 1

        print(f"\n日志统计:")
        print(f"  - ROI1数据不足警告: {roi1_insufficient_count} 次")
        print(f"  - ROI2独立检测: {roi2_detection_count} 次")
        print(f"  - 混合检测模式: {hybrid_detection_count} 次")

        if roi1_insufficient_count > 0 and roi2_detection_count == 0 and hybrid_detection_count == 0:
            print("\n⚠️  确认问题: 所有帧都因为ROI1数据不足而跳过检测")
            print("  建议修改配置: hybrid_detection.enabled = false")
            return True

        print("\n✓ 日志看起来正常")
        return False

    except Exception as e:
        print(f"❌ 读取日志文件失败: {e}")
        return None

def check_cache():
    """检查分析缓存"""
    print("\n" + "=" * 60)
    print("步骤3: 检查分析缓存")
    print("=" * 60)

    cache_dir = "export"
    if not os.path.exists(cache_dir):
        print(f"❌ 导出目录不存在: {cache_dir}")
        return

    cache_files = [f for f in os.listdir(cache_dir) if f.startswith('roi_analysis_cache_') and f.endswith('.jsonl')]
    if not cache_files:
        print(f"❌ 没有找到分析缓存文件")
        return

    cache_file = os.path.join(cache_dir, sorted(cache_files)[-1])
    print(f"✓ 读取最新缓存文件: {cache_file}")

    try:
        import json

        roi1_buffer_sizes = []
        frame_count = 0
        peak_frames = 0

        with open(cache_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if data.get('type') == 'frame':
                        frame_count += 1
                        # 检查是否有波峰
                        green_peaks = data.get('green_peaks', [])
                        red_peaks = data.get('red_peaks', [])
                        if green_peaks or red_peaks:
                            peak_frames += 1

                except:
                    pass

        print(f"\n缓存统计:")
        print(f"  - 总帧数: {frame_count}")
        print(f"  - 有波峰的帧数: {peak_frames}")

        if frame_count > 0 and peak_frames == 0:
            print("\n⚠️  确认问题: 所有帧都没有检测到波峰")
            return True
        elif peak_frames > 0:
            print(f"\n✓ 检测到 {peak_frames} 帧有波峰")
            return False

    except Exception as e:
        print(f"❌ 读取缓存文件失败: {e}")
        return None

def suggest_fix(config_info):
    """建议修复方案"""
    print("\n" + "=" * 60)
    print("建议的修复方案")
    print("=" * 60)

    if config_info and config_info.get('problem_detected'):
        print("\n方案1: 临时关闭混合检测（推荐）")
        print("  修改 simple_fem_config.json:")
        print("  {")
        print("    \"hybrid_detection\": {")
        print("      \"enabled\": false  // 改为 false")
        print("    }")
        print("  }")
        print("\n  然后重新运行: python -m refactor.main")

        print("\n方案2: 关闭ROI1检测")
        print("  修改 simple_fem_config.json:")
        print("  {")
        print("    \"roi1_peak_detection\": {")
        print("      \"enabled\": false  // 改为 false")
        print("    }")
        print("  }")
        print("\n  然后重新运行: python -m refactor.main")

        print("\n方案3: 等待ROI1缓冲区积累数据")
        print("  如果视频长度 > 100帧，ROI1缓冲区满后会自动使用混合检测")
        print("  但视频开始阶段的波峰会丢失")

        # 创建修复后的配置示例
        print("\n" + "=" * 60)
        print("是否要创建修复后的配置文件？(y/n)")
        # 注意：这里需要用户交互，暂时跳过

def main():
    """主函数"""
    print("SimpleFEM 波峰检测问题诊断工具")
    print("=" * 60)

    # 切换到项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)
    print(f"工作目录: {os.getcwd()}")

    # 执行检查
    config_info = check_config()
    problem_in_logs = check_logs()
    problem_in_cache = check_cache()

    # 给出建议
    if config_info:
        suggest_fix(config_info)

    print("\n" + "=" * 60)
    print("诊断完成")
    print("=" * 60)

if __name__ == '__main__':
    main()
