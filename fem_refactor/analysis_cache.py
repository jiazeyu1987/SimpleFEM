from __future__ import annotations

import json
import os
import platform
import sys
import uuid
from datetime import datetime
from typing import Any, Dict, Optional


def _json_default(obj: Any) -> Any:
    """json.dumps fallback for numpy / datetime / other non-serializable values."""
    try:
        import numpy as _np  # local import to avoid hard dependency in helper

        if isinstance(obj, (_np.integer, _np.floating)):
            return obj.item()
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
    except Exception:
        pass

    if isinstance(obj, datetime):
        return obj.isoformat()

    return str(obj)


class RoiAnalysisCache:
    """
    Write a lightweight per-frame cache to `export/` for later analysis.

    Format: JSONL (one JSON object per line), with `type` in {"meta","frame","session_end"}.
    """

    def __init__(self, export_dir: str, enabled: bool = True, flush_every: int = 50) -> None:
        self.export_dir = export_dir
        self.enabled = bool(enabled)
        self.flush_every = max(1, int(flush_every))
        self._fh: Optional[Any] = None
        self._path: Optional[str] = None
        self._run_id = uuid.uuid4().hex[:12]
        self._write_count = 0
        os.makedirs(self.export_dir, exist_ok=True)

    @property
    def path(self) -> Optional[str]:
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
        if not self.enabled:
            return

        self.close(reason="switch_session")

        safe_session = str(session_id or "unknown")
        filename = f"roi_analysis_cache_{safe_session}_{self._run_id}.jsonl"
        self._path = os.path.join(self.export_dir, filename)
        self._fh = open(self._path, "a", encoding="utf-8", newline="\n")
        self._write_count = 0

        meta: Dict[str, Any] = {
            "type": "meta",
            "cache_version": 1,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "session_id": safe_session,
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
        try:
            self._fh.flush()
        except Exception:
            pass

    def record_frame(self, payload: Dict[str, Any]) -> None:
        if not self.enabled or self._fh is None:
            return
        payload = dict(payload)
        payload.setdefault("type", "frame")
        self._write_line(payload)

    def close(self, reason: str = "normal") -> None:
        if not self.enabled or self._fh is None:
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
            self._fh.flush()
        except Exception:
            pass
        try:
            self._fh.close()
        except Exception:
            pass
        self._fh = None

    def _write_line(self, obj: Dict[str, Any]) -> None:
        if self._fh is None:
            return
        line = json.dumps(obj, ensure_ascii=False, default=_json_default)
        self._fh.write(line + "\n")
        self._write_count += 1
        if self._write_count % self.flush_every == 0:
            try:
                self._fh.flush()
            except Exception:
                pass

