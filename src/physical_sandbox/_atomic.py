"""Write a file so that it is either the old one or the whole new one, never half written."""
from __future__ import annotations

import os
from pathlib import Path


def write_text_atomic(path, text: str) -> None:
    """Write ``text`` (UTF-8) to a temporary file beside ``path``, then rename it over ``path``."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")  # beside it: the rename stays on one file system
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
