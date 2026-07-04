"""Use serial maps for SegmentAnyTree clustering when explicitly enabled."""

from __future__ import annotations

import multiprocessing
import multiprocessing.pool
import os
from typing import Any, Callable, Iterable


class SerialPool:
    """Pool subset used by SegmentAnyTree's HDBSCAN and mean-shift helpers."""

    def __init__(self, processes: int | None = None, *args: Any, **kwargs: Any) -> None:
        del processes, args, kwargs

    def __enter__(self) -> "SerialPool":
        return self

    def __exit__(self, *args: Any) -> None:
        del args

    def map(
        self,
        function: Callable[[Any], Any],
        iterable: Iterable[Any],
        chunksize: int | None = None,
    ) -> list[Any]:
        del chunksize
        return [function(item) for item in iterable]


if os.environ.get("SEGMENTANYTREE_SERIAL_POOL") == "1":
    multiprocessing.Pool = SerialPool
    multiprocessing.pool.Pool = SerialPool
