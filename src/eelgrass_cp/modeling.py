import json
import importlib
from typing import Dict, Tuple, Optional

import numpy as np
import torch


def load_model_and_config(emd_path: str, device: Optional[str] = None, chip_size: Optional[int] = None):
    """
    Load an ArcGIS model from an EMD file and return (model, config).

    Config fields:
      - chip_height, chip_width
      - min, max, scaled_mean, scaled_std (all np.float32)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(emd_path, "r") as f:
        data = json.load(f)

    model_name = data["ModelName"]
    ModelClass = getattr(importlib.import_module("arcgis.learn"), model_name)
    model_obj = ModelClass.from_model(data=None, emd_path=emd_path)
    model = model_obj.learn.model.to(device).eval()

    norm = data.get("NormalizationStats", None)
    if norm:
        cfg = {
            "chip_height": int(data.get("ImageHeight", chip_size or 0)),
            "chip_width": int(data.get("ImageWidth", chip_size or 0)),
            "min": np.array(norm["band_min_values"], np.float32),
            "max": np.array(norm["band_max_values"], np.float32),
            "scaled_mean": np.array(norm["scaled_mean_values"], np.float32),
            "scaled_std": np.array(norm["scaled_std_values"], np.float32),
        }
    else:
        # Reasonable defaults if missing.
        cfg = {
            "chip_height": int(data.get("ImageHeight", chip_size or 0)),
            "chip_width": int(data.get("ImageWidth", chip_size or 0)),
            "min": np.array([0.0, 0.0, 0.0], np.float32),
            "max": np.array([255.0, 255.0, 255.0], np.float32),
            "scaled_mean": np.array([0.485, 0.456, 0.406], np.float32),
            "scaled_std": np.array([0.229, 0.224, 0.225], np.float32),
        }

    return model, cfg


def norm_chip(chip: np.ndarray, cfg: Dict[str, np.ndarray]) -> np.ndarray:
    """Normalize chip (C,H,W) using per-EMD min/max and mean/std."""
    chip_hw_c = np.transpose(chip, (1, 2, 0)).astype(np.float32)
    denom = cfg["max"] - cfg["min"]
    denom = np.where(denom == 0, 1.0, denom)
    scaled = (chip_hw_c - cfg["min"]) / denom
    scaled = (scaled - cfg["scaled_mean"]) / np.where(cfg["scaled_std"] == 0, 1.0, cfg["scaled_std"])
    return np.transpose(scaled, (2, 0, 1))


@torch.no_grad()
def model_logits(model, chip_norm: np.ndarray, device: Optional[str] = None) -> np.ndarray:
    """Forward pass -> logits, returns (C,H,W) float32."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.from_numpy(chip_norm).float().unsqueeze(0).to(device)
    out = model(x)
    if isinstance(out, (list, tuple)):
        out = out[0]
    return out.squeeze(0).detach().cpu().numpy().astype(np.float32)


@torch.no_grad()
def model_probs(model, chip_norm: np.ndarray, device: Optional[str] = None) -> np.ndarray:
    """Forward pass -> softmax probs, returns (C,H,W) float32."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.from_numpy(chip_norm).float().unsqueeze(0).to(device)
    out = model(x)
    if isinstance(out, (list, tuple)):
        out = out[0]
    probs = torch.nn.functional.softmax(out, dim=1)
    return probs.squeeze(0).detach().cpu().numpy().astype(np.float32)


def softmax_vec(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.clip(np.sum(e), 1e-12, None)


def softmax_rows(logit_rows: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = logit_rows / max(T, 1e-8)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return (e / np.clip(e.sum(axis=1, keepdims=True), 1e-12, None)).astype(np.float32)


def softmax_3d(z: np.ndarray) -> np.ndarray:
    """Softmax over channel axis for (C,H,W)."""
    z = z - z.max(axis=0, keepdims=True)
    e = np.exp(z)
    return (e / np.clip(e.sum(axis=0, keepdims=True), 1e-12, None)).astype(np.float32)


def softmax_4d(z: np.ndarray) -> np.ndarray:
    """Softmax over channel axis for (M,C,H,W)."""
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return (e / np.clip(e.sum(axis=1, keepdims=True), 1e-12, None)).astype(np.float32)
