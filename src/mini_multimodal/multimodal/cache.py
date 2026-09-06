"""A small byte-bounded LRU cache shared by scene and QA dataset views."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

import numpy as np


T = TypeVar("T", bound=np.ndarray)


@dataclass(frozen=True, slots=True)
class CacheInfo:
    max_bytes: int
    current_bytes: int
    item_count: int
    hits: int
    misses: int
    evictions: int
    skipped_oversize: int


class ByteLRUCache:
    """Cache numpy arrays without ever exceeding ``max_bytes``."""

    def __init__(self, max_bytes: int = 64 * 1024 * 1024) -> None:
        if max_bytes < 0:
            raise ValueError("max_bytes must be non-negative")
        self.max_bytes = int(max_bytes)
        self._items: OrderedDict[str, np.ndarray] = OrderedDict()
        self._bytes = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._skipped_oversize = 0

    def get_or_load(self, key: str, loader: Callable[[], T]) -> T:
        if key in self._items:
            self._hits += 1
            value = self._items.pop(key)
            self._items[key] = value
            return value  # type: ignore[return-value]

        self._misses += 1
        value = loader()
        if not isinstance(value, np.ndarray):
            raise TypeError("ByteLRUCache only stores numpy arrays")
        value.setflags(write=False)
        size = int(value.nbytes)
        if self.max_bytes == 0 or size > self.max_bytes:
            self._skipped_oversize += 1
            return value

        while self._items and self._bytes + size > self.max_bytes:
            _, removed = self._items.popitem(last=False)
            self._bytes -= int(removed.nbytes)
            self._evictions += 1
        self._items[key] = value
        self._bytes += size
        return value

    def clear(self) -> None:
        self._items.clear()
        self._bytes = 0

    def info(self) -> CacheInfo:
        return CacheInfo(
            max_bytes=self.max_bytes,
            current_bytes=self._bytes,
            item_count=len(self._items),
            hits=self._hits,
            misses=self._misses,
            evictions=self._evictions,
            skipped_oversize=self._skipped_oversize,
        )
