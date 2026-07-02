from __future__ import annotations

import re
from pathlib import Path


def safe_storage_name(filename: str) -> str:
    stem = Path(filename).stem.lower()
    stem = re.sub(r"[^a-z0-9-]+", "-", stem).strip("-")
    return stem[:48] or "upload"

