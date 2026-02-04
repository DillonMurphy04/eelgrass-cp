from typing import Tuple

import numpy as np

# ---------------- Morton utilities (used in fixed spatial blocking) ----------------

def _part1by1(n: np.uint64) -> np.uint64:
    n &= 0xFFFFFFFF
    n = (n | (n << 16)) & 0x0000FFFF0000FFFF
    n = (n | (n << 8))  & 0x00FF00FF00FF00FF
    n = (n | (n << 4))  & 0x0F0F0F0F0F0F0F0F
    n = (n | (n << 2))  & 0x3333333333333333
    n = (n | (n << 1))  & 0x5555555555555555
    return n


def morton_code_2d(xi: np.ndarray, yi: np.ndarray) -> np.ndarray:
    xi = xi.astype(np.uint64)
    yi = yi.astype(np.uint64)
    x = np.vectorize(_part1by1, otypes=[np.uint64])(xi)
    y = np.vectorize(_part1by1, otypes=[np.uint64])(yi)
    return (x | (y << 1)).astype(np.uint64)


def quantize_to_grid(x: np.ndarray, y: np.ndarray, p: int) -> Tuple[np.ndarray, np.ndarray]:
    n = 1 << p
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    dx = max(xmax - xmin, 1e-12)
    dy = max(ymax - ymin, 1e-12)
    xi = np.floor((x - xmin) / dx * (n - 1)).astype(np.int64)
    yi = np.floor((y - ymin) / dy * (n - 1)).astype(np.int64)
    xi = np.clip(xi, 0, n - 1)
    yi = np.clip(yi, 0, n - 1)
    return xi, yi


def morton_order(x: np.ndarray, y: np.ndarray, p: int) -> Tuple[np.ndarray, np.ndarray]:
    xi, yi = quantize_to_grid(x, y, p)
    code = morton_code_2d(xi, yi)
    order = np.argsort(code, kind="mergesort")
    return order, code
