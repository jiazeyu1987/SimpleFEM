from __future__ import annotations

import logging
import logging.handlers
import os
from datetime import datetime

from .paths import get_base_dir

BASE_DIR = get_base_dir(__file__)


def setup_logging():
    """配置日志系统，输出到控制台和文件"""
    base_dir = BASE_DIR
    log_dir = os.path.join(base_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # 创建日志文件名（包含时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"simple_roi_daemon_{timestamp}.log")

    # 配置根日志记录器
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # 清除现有的处理器
    logger.handlers.clear()

    # 创建格式化器
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # 文件处理器（记录所有级别）
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 控制台处理器（只记录INFO及以上级别）
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 过滤第三方库的DEBUG日志（减少噪音）
    # PIL/Pillow 图片库
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)
    logging.getLogger("PIL.Image").setLevel(logging.WARNING)
    # OpenCV
    logging.getLogger("cv2").setLevel(logging.WARNING)
    # Matplotlib
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    # NumPy
    logging.getLogger("numpy").setLevel(logging.WARNING)

    logging.info(f"日志系统已启动，日志文件: {log_file}")
    return log_file


def setup_peak_logger() -> logging.Logger:
    """Create a logger that writes plain text lines and rotates daily."""
    logger = logging.getLogger("roi_peak_daemon")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    # Keep logs local to SimpleFEM project directory
    log_dir = os.path.join(BASE_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "roi_peak_daemon.log")

    handler = logging.handlers.TimedRotatingFileHandler(
        log_path,
        when="midnight",
        interval=1,
        backupCount=7,
        encoding="utf-8",
    )
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False

    return logger

