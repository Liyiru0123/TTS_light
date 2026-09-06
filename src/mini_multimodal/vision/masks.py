"""Binary-mask encoding and geometric measurements."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def encode_rle(mask: NDArray[np.bool_]) -> tuple[int, ...]:
    """Encode a 2-D mask as row-major, background-first uncompressed RLE."""

    if mask.ndim != 2:
        raise ValueError("RLE encoding requires a 2-D mask")
    flat = mask.astype(np.bool_, copy=False).reshape(-1)
    counts: list[int] = []
    current = False
    count = 0
    for value in flat:
        bit = bool(value)
        if bit == current:
            count += 1
        else:
            counts.append(count)
            count = 1
            current = bit
    counts.append(count)
    return tuple(counts)


def decode_rle(
    counts: tuple[int, ...] | list[int], height: int, width: int
) -> NDArray[np.bool_]:
    if height <= 0 or width <= 0:
        raise ValueError("Mask dimensions must be positive")
    if not counts or any(count < 0 for count in counts):
        raise ValueError("RLE counts must be non-empty and non-negative")
    if sum(counts) != height * width:
        raise ValueError("RLE counts do not match requested mask dimensions")
    flat = np.empty(height * width, dtype=np.bool_)
    offset = 0
    value = False
    for count in counts:
        flat[offset : offset + count] = value
        offset += count
        value = not value
    return flat.reshape(height, width)


def mask_bbox(mask: NDArray[np.bool_]) -> tuple[int, int, int, int] | None:
    if mask.ndim != 2:
        raise ValueError("Bounding boxes require a 2-D mask")
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1

