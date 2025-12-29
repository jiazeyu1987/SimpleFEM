"""
SimpleFEM Refactor - 运行所有测试

SimpleFEM Refactored Version
"""

import sys
import os
import unittest

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# 导入所有测试模块
from refactor.test.test_config_manager import TestConfigManager
from refactor.test.test_threshold_protection_manager import TestThresholdProtectionManager
from refactor.test.test_roi3_statistics import TestROI3Statistics
from refactor.test.test_hybrid_detection_manager import TestHybridDetectionManager
from refactor.test.test_green_line_manager import TestGreenLineManager
from refactor.test.test_analysis_cache_manager import TestAnalysisCacheManager
from refactor.test.test_data_export_manager import TestDataExportManager


def create_test_suite():
    """创建测试套件"""
    suite = unittest.TestSuite()

    # 添加所有测试类
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestConfigManager))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestThresholdProtectionManager))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestROI3Statistics))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestHybridDetectionManager))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestGreenLineManager))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestAnalysisCacheManager))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestDataExportManager))

    return suite


def run_tests(verbosity=2):
    """
    运行所有测试

    Args:
        verbosity: 详细程度 (0=静默, 1=正常, 2=详细)

    Returns:
        测试是否成功
    """
    suite = create_test_suite()
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)

    # 返回是否全部通过
    return result.wasSuccessful()


def main():
    """主函数"""
    print("=" * 70)
    print("SimpleFEM Refactor - 测试套件")
    print("=" * 70)
    print()

    # 运行测试
    success = run_tests(verbosity=2)

    print()
    print("=" * 70)
    if success:
        print("[PASS] All tests passed!")
    else:
        print("[FAIL] Some tests failed")
    print("=" * 70)

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
