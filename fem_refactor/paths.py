from __future__ import annotations

import os
import sys


def get_base_dir(file_path: str) -> str:
    """
    Resolve project base directory for both source (.py) and frozen (.exe) modes.

    - Frozen: use sys.executable directory
    - Source: use the directory of the caller file, but if it is under `fem_refactor/`,
      treat the parent directory as the project root.
    """
    if getattr(sys, "frozen", False) and hasattr(sys, "executable"):
        return os.path.dirname(os.path.abspath(sys.executable))

    this_dir = os.path.dirname(os.path.abspath(file_path))
    if os.path.basename(this_dir).lower() == "fem_refactor":
        return os.path.dirname(this_dir)
    return this_dir

