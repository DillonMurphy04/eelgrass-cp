"""
Deep ensemble point inference on 2022 points with optional chip visualizations.
Outputs: per-point cache CSV/NPZ and optional figures in OUT_DIR.
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
import torch
import fiona
from shapely.geometry import shape
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from src.eelgrass_cp import load_model_and_config, norm_chip, model_probs
from src.eelgrass_cp import safe_read_chip, ensure_uint8_rgb

# -------------------------------------------------------
# User Options
# -------------------------------------------------------
SAVE_VISUALIZATIONS = True
VISUALIZE_EVERY = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Data/paths
RASTER_PATH = r"D:\2022 Data\Morro Bay Eelgrass AI - Rasterized Imagery\2022_raster_final.tif"
POINTS_SHP = r"D:\2022 Data\2022points\2022points.shp"
OUT_DIR = r"D:\2022 Data\point_chips_validation_ensemble"

# Chip & label config
CHIP_SIZE = 448
GT_FIELD = "GROUND_TRU"
TARGET_CLASS = 1

# EMD model paths (keys are short names used in CSV columns)
MODEL_EMDS = {
    "unet_4yr": r"D:\4yr_Unet_model\models\checkpoint_2025-09-19_22-47-55_epoch_11\checkpoint_2025-09-19_22-47-55_epoch_11.emd",
    "samlora_4yr": r"D:\sam_lora_model\samlora_4_year\models\checkpoint_2025-06-11_18-24-03_epoch_18\checkpoint_2025-06-11_18-24-03_epoch_18.emd",
    "samlora_2021": r"D:\sam_lora_model_2021\models\checkpoint_2025-07-30_18-52-20_epoch_18\checkpoint_2025-07-30_18-52-20_epoch_18.emd",
    "deeplab_2021": r"D:\DeepLab_2021\DeepLab_2021.emd",
    "unet_2021": r"D:\2021 Unet\checkpoint_2025-06-25_12-39-26_epoch_11.emd",
    "deeplab_4yr": r"D:\deeplab_4_year\deeplab_4_year.emd",
}

os.makedirs(OUT_DIR, exist_ok=True)
FIG_DIR = os.path.join(OUT_DIR, "figs")
os.makedirs(FIG_DIR, exist_ok=True)


def plot_four_panel(fig_path, rgb, mean_prob, var_map, hard_mask, center_xy):
    """Save a 2x2 visualization."""
    cy, cx = center_xy
    plt.figure(figsize=(12, 10))

    ax1 = plt.subplot(2, 2, 1)
    ax1.imshow(rgb)
    ax1.plot(cx, cy, "ro", markersize=5)
    ax1.set_title("Raw Chip (+ point)")
    ax1.axis("off")

    ax2 = plt.subplot(2, 2, 2)
    im2 = ax2.imshow(mean_prob, vmin=0.0, vmax=1.0)
    ax2.plot(cx, cy, "wo", markersize=5)
    ax2.set_title("Ensemble Mean Probability (TARGET_CLASS)")
    ax2.axis("off")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    ax3 = plt.subplot(2, 2, 3)
    vmax_var = max(1e-6, float(np.percentile(var_map, 99.5)))
    im3 = ax3.imshow(var_map, vmin=0.0, vmax=vmax_var)
    ax3.plot(cx, cy, "wo", markersize=5)
    ax3.set_title("Ensemble Variance (TARGET_CLASS)")
    ax3.axis("off")
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    ax4 = plt.subplot(2, 2, 4)
    ax4.imshow(rgb)
    ax4.imshow(hard_mask.astype(np.uint8) * 255, alpha=0.4)
    ax4.plot(cx, cy, "wo", markersize=5)
    ax4.set_title("Hard Segmentation (argmax of ensemble mean)")
    ax4.axis("off")

    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()


print("Loading models...")
ensemble = {}
for key, emd in MODEL_EMDS.items():
    try:
        model, config = load_model_and_config(emd, device=DEVICE, chip_size=CHIP_SIZE)
        ensemble[key] = {"model": model, "config": config}
        print(f"  Loaded {key}")
    except Exception as e:
        warnings.warn(f"Failed to load {key} ({emd}): {e}")

if len(ensemble) == 0:
    raise RuntimeError("No models loaded; please check EMD paths.")

print("Reading points...")
with fiona.open(POINTS_SHP, "r") as src:
    points = [(shape(feat["geometry"]), feat["properties"]) for feat in src]
print(f"  Total points: {len(points)}")

print("Running deep ensemble inference on point chips...")
results = []

with rasterio.open(RASTER_PATH) as ds:
    for i, (geom, props) in enumerate(points):
        x, y = geom.x, geom.y
        row, col = ds.index(x, y)

        chip = safe_read_chip(ds, row, col, CHIP_SIZE)
        if chip is None:
            print(f"Point {i} near edge or failed read, skipping")
            continue

        per_model_center_probs = {}
        per_model_probmaps = []
        num_classes = None

        for key, pack in ensemble.items():
            try:
                chip_norm = norm_chip(chip, pack["config"])
                probs = model_probs(pack["model"], chip_norm, device=DEVICE)
                if num_classes is None:
                    num_classes = probs.shape[0]
                else:
                    if probs.shape[0] != num_classes:
                        C = probs.shape[0]
                        if C < num_classes:
                            pad = np.zeros((num_classes - C, probs.shape[1], probs.shape[2]), dtype=probs.dtype)
                            probs = np.concatenate([probs, pad], axis=0)
                        else:
                            probs = probs[:num_classes]
                per_model_probmaps.append(probs)

                cy, cx = CHIP_SIZE // 2, CHIP_SIZE // 2
                p_center = probs[min(TARGET_CLASS, probs.shape[0] - 1), cy, cx]
                per_model_center_probs[key] = float(p_center)
            except Exception as e:
                warnings.warn(f"Inference failed for model {key} at point {i}: {e}")

        if len(per_model_probmaps) == 0:
            print(f"No model predictions for point {i}, skipping")
            continue

        stack = np.stack(per_model_probmaps, axis=0)
        mean_probs = np.mean(stack, axis=0)
        tc = min(TARGET_CLASS, mean_probs.shape[0] - 1)
        var_map = np.var(stack[:, tc, :, :], axis=0)
        hard_mask = np.argmax(mean_probs, axis=0).astype(np.int32)

        cy, cx = CHIP_SIZE // 2, CHIP_SIZE // 2
        center_mean = float(mean_probs[tc, cy, cx])
        center_var = float(var_map[cy, cx])
        ensemble_pred_center = int(hard_mask[cy, cx])

        gt_val = props.get(GT_FIELD, None)
        gt_int = int(gt_val) if gt_val is not None else None

        fig_path = ""
        if SAVE_VISUALIZATIONS and (i % VISUALIZE_EVERY == 0):
            rgb = ensure_uint8_rgb(chip)
            fig_path = os.path.join(FIG_DIR, f"point_{i:04d}.png")
            plot_four_panel(
                fig_path,
                rgb=rgb,
                mean_prob=mean_probs[tc],
                var_map=var_map,
                hard_mask=hard_mask,
                center_xy=(cy, cx),
            )

        row_dict = {
            "id": i,
            "x": x,
            "y": y,
            "gt": gt_int,
            "ensemble_pred_center": ensemble_pred_center,
            "mean_prob_center": center_mean,
            "var_center": center_var,
            "fig_path": fig_path,
        }
        for key, p in per_model_center_probs.items():
            row_dict[f"{key}_p_center"] = p

        results.append(row_dict)


df = pd.DataFrame(results)

csv_path = os.path.join(OUT_DIR, "point_predictions_ensemble.csv")
df.to_csv(csv_path, index=False)

acc = None
if "gt" in df.columns:
    df_valid = df[~df["gt"].isna()].copy()
    if len(df_valid) > 0:
        y_true = df_valid["gt"].astype(int).values
        y_pred = df_valid["ensemble_pred_center"].astype(int).values
        acc = accuracy_score(y_true, y_pred)
        print(f"Done. Saved {len(results)} results to {OUT_DIR}")
        print(f"Center-pixel Accuracy = {acc:.4f}  (rows with GT: {len(df_valid)})")
    else:
        print(f"Done. Saved {len(results)} results to {OUT_DIR}")
        print("No rows with numeric GT found; accuracy not computed.")
else:
    print(f"Done. Saved {len(results)} results to {OUT_DIR}")
    print("No 'gt' column found; accuracy not computed.")
