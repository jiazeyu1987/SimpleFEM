"""
SimpleFEM 重构版本 - 主入口文件

SimpleFEM Refactored Version

用法:
    python -m refactor.main
    或者
    python refactor/main.py
"""

import sys
import os

# 添加父目录到路径以导入原始模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from refactor.orchestrator import Orchestrator


def main():
    """主函数"""
    print("=" * 60)
    print("SimpleFEM 重构版本启动")
    print("=" * 60)

    try:
        # 创建编排器并运行
        orchestrator = Orchestrator()
        orchestrator.run()
        orchestrator.close()

        print("\n" + "=" * 60)
        print("处理完成")
        print("=" * 60)

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
