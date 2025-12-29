#!/usr/bin/env python3
"""
诊断最基本的问题
"""
import os
import json
import cv2

def check_config():
    """检查配置文件"""
    print("配置文件检查:")
    print("=" * 30)

    if not os.path.exists('simple_fem_config.json'):
        print("❌ simple_fem_config.json 不存在")
        return False

    try:
        with open('simple_fem_config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)

        processing_mode = config.get("processing_mode", "")
        video_config = config.get("video_processing", {})
        video_path = video_config.get("video_path", "")

        print(f"processing_mode: {processing_mode}")
        print(f"video_path: {video_path}")

        if processing_mode != "video":
            print("❌ processing_mode 不是 'video'")
            return False

        if not video_path:
            print("❌ video_path 为空")
            return False

        return True

    except Exception as e:
        print(f"❌ 配置读取错误: {e}")
        return False

def check_video_files():
    """检查视频文件"""
    print("\n视频文件检查:")
    print("=" * 30)

    try:
        with open('simple_fem_config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)

        video_path = config.get("video_processing", {}).get("video_path", "")

        if os.path.isfile(video_path):
            print(f"✅ 单个视频文件: {video_path}")
            return [video_path]

        elif os.path.isdir(video_path):
            print(f"✅ 视频文件夹: {video_path}")

            # 查找视频文件
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
            video_files = []

            for filename in os.listdir(video_path):
                if any(filename.lower().endswith(ext) for ext in video_extensions):
                    video_files.append(os.path.join(video_path, filename))

            print(f"发现 {len(video_files)} 个视频文件:")
            for i, video_file in enumerate(video_files[:5], 1):  # 只显示前5个
                size_mb = os.path.getsize(video_file) / (1024*1024)
                print(f"  {i}. {os.path.basename(video_file)} ({size_mb:.1f} MB)")

            if len(video_files) > 5:
                print(f"  ... 还有 {len(video_files) - 5} 个文件")

            return video_files

        else:
            print(f"❌ 路径不存在: {video_path}")
            return []

    except Exception as e:
        print(f"❌ 视频文件检查错误: {e}")
        return []

def test_video_capture(video_files):
    """测试视频捕获"""
    if not video_files:
        print("\n❌ 没有视频文件可测试")
        return

    print(f"\n视频捕获测试 (测试第一个视频):")
    print("=" * 30)

    video_path = video_files[0]
    print(f"测试视频: {os.path.basename(video_path)}")

    try:
        # 测试打开
        video_cap = cv2.VideoCapture(video_path)
        if not video_cap.isOpened():
            print("❌ 无法打开视频")
            return

        print("✅ 视频成功打开")

        # 获取视频信息
        fps = video_cap.get(cv2.CAP_PROP_FPS)
        frame_count = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        width = video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        print(f"FPS: {fps}")
        print(f"总帧数: {frame_count}")
        print(f"分辨率: {width}x{height}")

        # 测试读取第一帧
        ret, frame = video_cap.read()
        if ret:
            print("✅ 成功读取第一帧")
            print(f"帧数据类型: {type(frame)}")
            print(f"帧尺寸: {frame.shape}")
        else:
            print("❌ 无法读取第一帧")

        video_cap.release()

    except Exception as e:
        print(f"❌ 视频捕获测试错误: {e}")

if __name__ == "__main__":
    print("SimpleFEM 基础诊断")
    print("=" * 50)

    # 检查配置
    config_ok = check_config()

    if config_ok:
        # 检查视频文件
        video_files = check_video_files()

        # 测试视频捕获
        test_video_capture(video_files)

    print("\n" + "=" * 50)
    print("诊断完成！")
    print("如果所有检查都通过，问题可能在于:")
    print("1. 依赖库问题 (opencv-python, PIL, matplotlib)")
    print("2. 权限问题 (写入权限)")
    print("3. 运行时的异常被静默忽略")