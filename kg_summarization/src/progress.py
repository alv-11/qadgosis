from __future__ import annotations

import os
import sys
from typing import Any

from tqdm.auto import tqdm


def progress(iterable=None, **kwargs: Any):
    """Create a consistent tqdm progress bar for terminal runs."""
    kwargs.setdefault("ascii", True)
    kwargs.setdefault("dynamic_ncols", True)
    kwargs.setdefault(
        "disable",
        os.environ.get("TQDM_DISABLE") == "1" or not sys.stderr.isatty(),
    )
    return tqdm(iterable, **kwargs)


def progress_write(*args: Any, sep: str = " ", end: str = "\n") -> None:
    message = sep.join(str(arg) for arg in args)
    tqdm.write(message, end=end)


def format_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.2f}s"

    minutes, remaining_seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {remaining_seconds:.1f}s"

    hours, remaining_minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(remaining_minutes)}m"


def format_bytes(num_bytes: int | float | None) -> str:
    if num_bytes is None:
        return "-"

    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            if unit == "B":
                return f"{int(value)}{unit}"
            return f"{value:.1f}{unit}"
        value /= 1024