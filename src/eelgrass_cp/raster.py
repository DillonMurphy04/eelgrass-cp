from typing import List, Tuple, Optional
import os

import numpy as np
import fiona
from shapely.geometry import shape
from rasterio.windows import Window


def safe_read_chip(ds, row: int, col: int, size: int):
    half = size // 2
    win = Window(col - half, row - half, size, size)
    chip = ds.read(window=win)
    if chip.shape[1] < size or chip.shape[2] < size:
        return None
    return chip


def ensure_uint8_rgb(chip: np.ndarray) -> np.ndarray:
    """Return an RGB uint8 array (H,W,3) for visualization."""
    c = min(3, chip.shape[0])
    rgb = np.moveaxis(chip[:c], 0, -1).astype(np.float32)
    vmin = np.percentile(rgb, 1)
    vmax = np.percentile(rgb, 99)
    if vmax <= vmin:
        vmax = vmin + 1.0
    rgb = np.clip((rgb - vmin) / (vmax - vmin), 0, 1)
    out = (rgb * 255.0).astype(np.uint8)
    if out.shape[2] < 3:
        out = np.repeat(out, 3, axis=2)
    return out


def read_map_pairs(map_txt: str, base_dir: Optional[str] = None) -> List[Tuple[str, str]]:
    pairs = []
    with open(map_txt, "r") as f:
        for line in f:
            if not line.strip():
                continue
            img_rel, lbl_rel = line.strip().split()
            if base_dir:
                img_rel = os.path.join(base_dir, img_rel)
                lbl_rel = os.path.join(base_dir, lbl_rel)
            pairs.append((img_rel, lbl_rel))
    return pairs


def read_points_shp(points_shp: str, max_points: Optional[int] = None):
    """
    Read a shapefile of points and return a list of (geometry, properties).
    """
    out = []
    with fiona.open(points_shp, "r") as src:
        for i, feat in enumerate(src):
            if max_points is not None and i >= max_points:
                break
            out.append((shape(feat["geometry"]), feat["properties"]))
    return out
