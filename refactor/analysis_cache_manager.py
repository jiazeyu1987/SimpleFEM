"""
分析缓存管理器 - JSONL格式缓存管理

SimpleFEM Refactored Version
"""

import json
import os
import platform
import sys
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from refactor.config_manager import ConfigManager


class AnalysisCacheManager:
    """
    分析缓存管理器

    功能:
    - 写入JSONL格式的每帧分析缓存
    - 支持会话元数据
    - 自动刷新和关闭处理
    """

    def __init__(self, config: ConfigManager, export_dir: str):
        """
        初始化分析缓存管理器

        Args:
            config: 配置管理器
            export_dir: 导出目录
        """
        self._config = config
        self._export_dir = export_dir
        self._enabled = config.analysis_cache_enabled
        self._flush_every = config.analysis_cache_flush_every

        self._fh: Optional[Any] = None
        self._path: Optional[str] = None
        self._run_id = uuid.uuid4().hex[:12]
        self._write_count = 0
        self._current_session_id: Optional[str] = None

        os.makedirs(self._export_dir, exist_ok=True)

    @property
    def path(self) -> Optional[str]:
        """缓存文件路径"""
        return self._path

    def start_session(
        self,
        session_id: str,
        *,
        processing_mode: str,
        video_path: Optional[str],
        config: Dict[str, Any],
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        开始新会话

        Args:
            session_id: 会话ID
            processing_mode: 处理模式
            video_path: 视频路径
            config: 配置字典
            extra_meta: 额外元数据
        """
        if not self._enabled:
            return

        self.close(reason="switch_session")

        self._current_session_id = str(session_id or "unknown")
        filename = f"roi_analysis_cache_{self._current_session_id}_{self._run_id}.jsonl"
        self._path = os.path.join(self._export_dir, filename)
        self._fh = open(self._path, "a", encoding="utf-8", newline="\n")
        self._write_count = 0

        meta: Dict[str, Any] = {
            "type": "meta",
            "cache_version": 1,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "session_id": self._current_session_id,
            "processing_mode": processing_mode,
            "video_path": video_path,
            "host": {
                "platform": platform.platform(),
                "python": sys.version.split()[0],
            },
            "config": config,
        }
        if extra_meta:
            meta["extra"] = extra_meta

        self._write_line(meta)
        self._flush()

    def record_frame(self, payload: Dict[str, Any]) -> None:
        """
        记录帧数据

        Args:
            payload: 帧数据字典
        """
        if not self._enabled or self._fh is None:
            return

        payload = dict(payload)
        payload.setdefault("type", "frame")
        self._write_line(payload)

    def close(self, reason: str = "normal") -> None:
        """
        关闭缓存

        Args:
            reason: 关闭原因
        """
        if not self._enabled or self._fh is None:
            self._fh = None
            self._path = self._path  # keep last path for reference
            return

        try:
            self._write_line(
                {
                    "type": "session_end",
                    "ended_at": datetime.now().isoformat(timespec="seconds"),
                    "reason": reason,
                }
            )
            self._flush()
        except Exception:
            pass

        try:
            self._fh.close()
        except Exception:
            pass

        self._fh = None

    def _write_line(self, obj: Dict[str, Any]) -> None:
        """写入一行JSON"""
        if self._fh is None:
            return

        line = json.dumps(obj, ensure_ascii=False, default=self._json_default)
        self._fh.write(line + "\n")
        self._write_count += 1

        if self._write_count % self._flush_every == 0:
            self._flush()

    def _flush(self) -> None:
        """刷新文件缓冲"""
        if self._fh is None:
            return
        try:
            self._fh.flush()
        except Exception:
            pass

    @staticmethod
    def _json_default(obj: Any) -> Any:
        """JSON序列化默认处理"""
        try:
            import numpy as _np

            if isinstance(obj, (_np.integer, _np.floating)):
                return obj.item()
            if isinstance(obj, _np.ndarray):
                return obj.tolist()
        except Exception:
            pass

        if isinstance(obj, datetime):
            return obj.isoformat()

        return str(obj)

    @property
    def current_session_id(self) -> Optional[str]:
        """当前会话ID"""
        return self._current_session_id
