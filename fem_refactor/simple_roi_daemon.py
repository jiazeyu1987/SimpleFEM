"""Entry point for the SimpleFEM ROI daemon.

The full implementation lives in `fem_refactor/`.
"""

from __future__ import annotations

import os
import sys


def _ensure_project_root_on_sys_path() -> None:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


_ensure_project_root_on_sys_path()

from fem_refactor.orchestrator import run_daemon  # noqa: E402


def main() -> None:
    run_daemon()


if __name__ == "__main__":
    main()
