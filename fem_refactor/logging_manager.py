from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from typing import Any, Optional

from .paths import get_base_dir

BASE_DIR = get_base_dir(__file__)


_MASTER_LOGGING_ENABLED: bool = True
_ORIG_STDOUT: Optional[Any] = None
_ORIG_STDERR: Optional[Any] = None
_DEVNULL: Optional[Any] = None


def _parse_bool_env(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return None


def resolve_master_logging_enabled(config: Optional[dict] = None) -> bool:
    env_override = _parse_bool_env(os.getenv("SIMPLEFEM_LOGGING_ENABLED"))
    if env_override is not None:
        return env_override

    if isinstance(config, dict):
        logging_cfg = config.get("logging", {})
        if isinstance(logging_cfg, dict) and "enabled" in logging_cfg:
            return bool(logging_cfg.get("enabled", True))

    return True


def set_master_logging_enabled(enabled: bool) -> None:
    global _MASTER_LOGGING_ENABLED, _ORIG_STDOUT, _ORIG_STDERR, _DEVNULL

    enabled = bool(enabled)
    _MASTER_LOGGING_ENABLED = enabled

    if enabled:
        if _ORIG_STDOUT is not None:
            sys.stdout = _ORIG_STDOUT
        if _ORIG_STDERR is not None:
            sys.stderr = _ORIG_STDERR
        if _DEVNULL is not None:
            try:
                _DEVNULL.close()
            except Exception:
                pass
        _ORIG_STDOUT = None
        _ORIG_STDERR = None
        _DEVNULL = None
        logging.getLogger().disabled = False
        logging.disable(logging.NOTSET)
        return

    if _ORIG_STDOUT is None:
        _ORIG_STDOUT = sys.stdout
    if _ORIG_STDERR is None:
        _ORIG_STDERR = sys.stderr
    if _DEVNULL is None:
        _DEVNULL = open(os.devnull, "w", encoding="utf-8", errors="ignore")

    sys.stdout = _DEVNULL
    sys.stderr = _DEVNULL
    logging.disable(logging.CRITICAL)


def is_master_logging_enabled() -> bool:
    return bool(_MASTER_LOGGING_ENABLED)


def setup_logging(*, enabled: Optional[bool] = None, config: Optional[dict] = None) -> str:
    """配置日志系统，输出到控制台和文件"""
    if enabled is None:
        enabled = resolve_master_logging_enabled(config)
    set_master_logging_enabled(enabled)
    if not enabled:
        root = logging.getLogger()
        root.handlers.clear()
        root.disabled = True
        return ""
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


def setup_peak_logger(*, enabled: Optional[bool] = None) -> logging.Logger:
    """Create a logger that writes plain text lines and rotates daily."""
    if enabled is None:
        enabled = is_master_logging_enabled()

    logger = logging.getLogger("roi_peak_daemon")
    if not enabled:
        logger.handlers.clear()
        logger.addHandler(logging.NullHandler())
        logger.propagate = False
        logger.disabled = True
        return logger

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
