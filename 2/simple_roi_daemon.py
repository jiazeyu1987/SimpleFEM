"""Entry point wrapper for the SimpleFEM ROI daemon.

The full implementation has been refactored into `fem_refactor/`.
"""

from __future__ import annotations

from fem_refactor.orchestrator import run_daemon


if __name__ == "__main__":
    run_daemon()
