from __future__ import annotations

import json
import os
from typing import Dict

from .paths import get_base_dir

BASE_DIR = get_base_dir(__file__)


def load_fem_config() -> Dict:
    """Load simple_fem_config.json from backend/app (only fields needed by this script)."""
    config_path = os.path.join(BASE_DIR, "simple_fem_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

