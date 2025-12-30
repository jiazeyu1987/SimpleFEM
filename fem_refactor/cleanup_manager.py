from __future__ import annotations

import logging
import os
import stat
import shutil

from .config_loader import load_fem_config
from .paths import get_base_dir

BASE_DIR = get_base_dir(__file__)


def cleanup_directories():
    """根据配置文件清理指定文件夹下的所有内容"""
    try:
        config = load_fem_config()
        cleanup_config = config.get("startup_cleanup", {})

        # 检查是否启用清理功能
        if not cleanup_config.get("enabled", True):
            logging.info("启动时清理功能已禁用（配置文件中 startup_cleanup.enabled = false）")
            return

        external_base_dir = os.path.join(BASE_DIR, "fem_refactor", "external")
        os.makedirs(external_base_dir, exist_ok=True)

        export_dir = os.path.join(BASE_DIR, "export")
        external_export_dir = os.path.join(external_base_dir, "export")
        tmp_dir = os.path.join(external_base_dir, "tmp")
        logs_dir = os.path.join(external_base_dir, "logs")

        # 每次启动都清理 export + tmp（用户诉求：固定行为），并且不清理 logs（避免日志被清空）
        directories_to_clean = ["export", "tmp"]

        # 检查各个目录的清理开关
        cleanup_switches = {
            "export": cleanup_config.get("cleanup_export", True),
            "tmp": cleanup_config.get("cleanup_tmp", True),
            "logs": False,
        }

        cleaned_count = 0

        logging.info("开始启动时清理...")

        def _on_rm_error(func, path, exc_info):
            try:
                os.chmod(path, stat.S_IWRITE)
                func(path)
            except Exception:
                pass

        def _clean_dir(dir_path: str, label: str) -> None:
            nonlocal cleaned_count
            if not os.path.exists(dir_path):
                logging.info(f"  文件夹不存在，跳过: {label}")
                return

            try:
                items_to_delete = os.listdir(dir_path)
                if not items_to_delete:
                    logging.info(f"  目录 {label} 为空，无需清理")
                    return

                logging.info(f"清理文件夹: {dir_path}（包含 {len(items_to_delete)} 个项目）")

                deleted_files = 0
                deleted_dirs = 0
                for item_name in items_to_delete:
                    item_path = os.path.join(dir_path, item_name)
                    try:
                        if os.path.isfile(item_path) or os.path.islink(item_path):
                            try:
                                os.chmod(item_path, stat.S_IWRITE)
                            except Exception:
                                pass
                            os.remove(item_path)
                            logging.debug(f"  删除文件: {item_name}")
                            deleted_files += 1
                        elif os.path.isdir(item_path):
                            shutil.rmtree(item_path, onerror=_on_rm_error)
                            logging.debug(f"  删除文件夹: {item_name}")
                            deleted_dirs += 1
                    except Exception as item_error:
                        logging.warning(f"  删除失败 {item_name}: {item_error}")

                logging.info(f"  清理完成: {label}（删除 {deleted_files} 个文件，{deleted_dirs} 个文件夹）")
                cleaned_count += 1
            except Exception as e:
                logging.error(f"  清理文件夹失败: {e}")

        for dir_name in directories_to_clean:
            # 检查该目录是否被标记为可清理
            if dir_name not in cleanup_switches or not cleanup_switches[dir_name]:
                logging.info(f"跳过目录 {dir_name}（配置文件中已禁用）")
                continue

            if dir_name == "export":
                _clean_dir(export_dir, "export")
                _clean_dir(external_export_dir, "external/export")
                continue
            elif dir_name == "tmp":
                dir_path = tmp_dir
            elif dir_name == "logs":
                dir_path = logs_dir
            else:
                dir_path = os.path.join(BASE_DIR, dir_name)

            _clean_dir(dir_path, dir_name)

        if cleaned_count == 0:
            logging.info("没有需要清理的目录或所有目录都为空")
        else:
            logging.info(f"清理完成：共清理了 {cleaned_count} 个目录")

    except Exception as e:
        logging.error(f"读取清理配置时发生错误: {e}")
        # 如果配置读取失败，使用默认行为（不清理）
        logging.info("由于配置读取失败，跳过启动时清理")
