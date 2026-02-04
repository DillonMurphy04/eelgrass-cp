"""
2022 point evaluation using a calibration summary (vanilla + linear).
Outputs: per-point CSVs, summary JSON, and optional visualizations.
"""

import os
import json
import warnings
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
import torch
import fiona
from shapely.geometry import shape
import pandas as pd
import matplotlib.pyplot as plt

from eelgrass_cp import load_model_and_config, norm_chip, model_logits
from eelgrass_cp import softmax_vec
from eelgrass_cp import ensure_uint8_rgb, safe_read_chip

# ----------------------------- User Options -----------------------------
SAVE_VISUALIZATIONS = True
VISUALIZE_EVERY = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RASTER_PATH = r"D:\2022 Data\Morro Bay Eelgrass AI - Rasterized Imagery\2022_raster_final.tif"
POINTS_SHP = r"D:\2022 Data\2022points\2022points.shp"
OUT_DIR = r"D:\2022 Data\cp_eval_2022_points"
os.makedirs(OUT_DIR, exist_ok=True)
FIG_DIR = os.path.join(OUT_DIR, "figs")
os.makedirs(FIG_DIR, exist_ok=True)

CALIB_SUMMARY_JSON = r"D:\calibration\summary.json"

CHIP_SIZE = 448
GT_FIELD = "GROUND_TRU"
N_CLASSES = 2

# Used when calibration meta provides a per-lambda linear table.
LAM_LINEAR = 3.0

MODEL_EMDS = {
    "unet_2020": r"D:\Unet_model\Unet2020\Unet2020.emd",
    "samlora_2020": r"D:\sam_lora_model\sam_lora_model_2020\sam_lora_model_2020.emd",
    "unet_2021": r"D:\2021 Unet\checkpoint_2025-06-25_12-39-26_epoch_11.emd",
    "deeplab_2021": r"D:\DeepLab_2021\DeepLab_2021.emd",
    "samlora_2021": r"D:\sam_lora_model_2021\models\checkpoint_2025-07-30_18-52-20_epoch_18\checkpoint_2025-07-30_18-52-20_epoch_18.emd",
}


def plot_four_panel(fig_path, rgb, mean_prob_tc, var_tc, hard_mask, center_xy):
    cy, cx = center_xy
    plt.figure(figsize=(12, 10))
    ax1 = plt.subplot(2, 2, 1)
    ax1.imshow(rgb)
    ax1.plot(cx, cy, "ro", ms=5)
    ax1.set_title("RGB")
    ax1.axis("off")

    ax2 = plt.subplot(2, 2, 2)
    im2 = ax2.imshow(mean_prob_tc, vmin=0, vmax=1)
    ax2.plot(cx, cy, "wo", ms=5)
    ax2.set_title("Mean p(class 1)")
    ax2.axis("off")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    vmax_var = max(1e-6, float(np.percentile(var_tc, 99.5)))
    ax3 = plt.subplot(2, 2, 3)
    im3 = ax3.imshow(var_tc, vmin=0, vmax=vmax_var)
    ax3.plot(cx, cy, "wo", ms=5)
    ax3.set_title("Variance (class 1)")
    ax3.axis("off")
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    ax4 = plt.subplot(2, 2, 4)
    ax4.imshow(rgb)
    ax4.imshow(hard_mask.astype(np.uint8) * 255, alpha=0.35)
    ax4.plot(cx, cy, "wo", ms=5)
    ax4.set_title("Hard Seg (argmax)")
    ax4.axis("off")

    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()


print("Loading calibration summary...")
with open(CALIB_SUMMARY_JSON, "r") as f:
    calib = json.load(f)

ALPHA = float(calib.get("alpha", 0.10))
T = float(calib.get("temperature_T", 1.0))
qhat_van = float(calib["pixel_level"]["vanilla"]["qhat"])

qhat_linear = None
lam_linear = None
adaptive = calib.get("pixel_level", {}).get("adaptive", {})
if "linear" in adaptive and isinstance(adaptive["linear"], dict):
    lam_linear = float(LAM_LINEAR)
    linear_pack = adaptive["linear"]
    for lam_str, pack in linear_pack.items():
        if abs(float(lam_str) - lam_linear) < 1e-8:
            qhat_linear = float(pack["qhat"])
            break
elif "qhat" in adaptive and "lambda" in adaptive:
    lam_linear = float(adaptive["lambda"])
    qhat_linear = float(adaptive["qhat"])

if qhat_linear is None or lam_linear is None:
    raise RuntimeError("Linear qhat/lambda not found in calibration summary.")

VAR_MIN = calib.get("variance_min", None)
VAR_MAX = calib.get("variance_max", None)

print(f"  alpha={ALPHA:.3f}  T={T:.3f}  qhat_van={qhat_van:.3f}  qhat_linear={qhat_linear:.3f}  lambda={lam_linear:.2f}")

print("Loading 2020+2021 ensemble models...")
ensemble = {}
for key, emd in MODEL_EMDS.items():
    try:
        model, cfg = load_model_and_config(emd, device=DEVICE)
        ensemble[key] = {"model": model, "config": cfg}
        print(f"  Loaded {key}")
    except Exception as e:
        warnings.warn(f"Failed to load {key}: {e}")
if not ensemble:
    raise RuntimeError("No models loaded")

print("Reading annotated points (2022)...")
with fiona.open(POINTS_SHP, "r") as src:
    points = [(shape(feat["geometry"]), feat["properties"]) for feat in src]
print(f"  Points: {len(points)}")

print("Running ensemble inference + CP scoring at points...")
rows = []
var_centers_list = []

with rasterio.open(RASTER_PATH) as ds:
    for i, (geom, props) in enumerate(points):
        x, y = geom.x, geom.y
        row, col = ds.index(x, y)

        chip = safe_read_chip(ds, row, col, CHIP_SIZE)
        if chip is None:
            warnings.warn(f"Point {i} near edge - skipped.")
            continue

        per_model_logits = []
        per_model_probs_c1 = []
        num_classes = None

        for key, pack in ensemble.items():
            try:
                chip_norm = norm_chip(chip, pack["config"])
                logits = model_logits(pack["model"], chip_norm, device=DEVICE)
                if num_classes is None:
                    num_classes = logits.shape[0]
                else:
                    if logits.shape[0] != num_classes:
                        C = logits.shape[0]
                        if C < num_classes:
                            pad = np.zeros((num_classes - C, logits.shape[1], logits.shape[2]), dtype=logits.dtype)
                            logits = np.concatenate([logits, pad], axis=0)
                        else:
                            logits = logits[:num_classes]
                per_model_logits.append(logits)

                probs = np.exp(logits - logits.max(axis=0, keepdims=True))
                probs = probs / np.clip(probs.sum(axis=0, keepdims=True), 1e-12, None)
                c1 = min(1, probs.shape[0] - 1)
                per_model_probs_c1.append(probs[c1])
            except Exception as e:
                warnings.warn(f"Inference failed for model {key} at point {i}: {e}")

        if len(per_model_logits) == 0:
            continue

        logits_stack = np.stack(per_model_logits, axis=0)
        mean_logits = np.mean(logits_stack, axis=0)
        c1_var_map = np.var(np.stack(per_model_probs_c1, 0), axis=0)

        cy, cx = CHIP_SIZE // 2, CHIP_SIZE // 2
        mean_logits_center = mean_logits[:, cy, cx]
        probs_center = softmax_vec(mean_logits_center / max(T, 1e-8))
        y_true = props.get(GT_FIELD, None)
        if y_true is None:
            continue
        y_true = int(y_true)

        hard_mask = np.argmax(
            np.exp(mean_logits - mean_logits.max(axis=0, keepdims=True)) /
            np.clip(np.exp(mean_logits - mean_logits.max(axis=0, keepdims=True)).sum(axis=0, keepdims=True), 1e-12, None),
            axis=0
        ).astype(np.int32)

        V_center = float(c1_var_map[cy, cx])
        var_centers_list.append(V_center)

        fig_path = ""
        if SAVE_VISUALIZATIONS and (i % VISUALIZE_EVERY == 0):
            rgb = ensure_uint8_rgb(chip)
            soft_all = np.exp(mean_logits - mean_logits.max(axis=0, keepdims=True))
            soft_all = soft_all / np.clip(soft_all.sum(axis=0, keepdims=True), 1e-12, None)
            mean_prob_c1 = soft_all[min(1, soft_all.shape[0] - 1)]
            fig_path = os.path.join(FIG_DIR, f"pt_{i:04d}.png")
            plot_four_panel(fig_path, rgb, mean_prob_c1, c1_var_map, hard_mask, (cy, cx))

        rows.append({
            "id": i,
            "x": x,
            "y": y,
            "gt": y_true,
            "p0_center": float(probs_center[0]),
            "p1_center": float(probs_center[min(1, len(probs_center) - 1)]),
            "V_center": V_center,
            "fig_path": fig_path,
        })

# Per-point frame

df = pd.DataFrame(rows)
csv_points = os.path.join(OUT_DIR, "points_with_scores.csv")
df.to_csv(csv_points, index=False)
print(f"Saved per-point scores: {csv_points} ({len(df)} rows)")

if VAR_MIN is None or VAR_MAX is None:
    VAR_MIN = float(np.min(df["V_center"])) if len(df) else 0.0
    VAR_MAX = float(np.max(df["V_center"])) if len(df) else 1.0
    warnings.warn(
        "No variance_min/max in calibration summary; "
        f"using 2022 min/max for normalization: [{VAR_MIN:.3e}, {VAR_MAX:.3e}]"
    )

Vn = (df["V_center"].values - VAR_MIN) / max(VAR_MAX - VAR_MIN, 1e-8)
Vn = np.clip(Vn, 0.0, 1.0)

# Scores and coverage

y = df["gt"].astype(int).values
p1 = df["p1_center"].values
p0 = df["p0_center"].values
p_true = np.where(y == 1, p1, p0)

scores_van = 1.0 - p_true
scores_linear = scores_van / (1.0 + lam_linear * Vn)

covered_van = (scores_van <= qhat_van)
covered_linear = (scores_linear <= qhat_linear)

coverage_van = float(np.mean(covered_van)) if len(covered_van) else float("nan")
coverage_linear = float(np.mean(covered_linear)) if len(covered_linear) else float("nan")

# Set composition (binary)

def set_composition_row(p0, p1, q, vnorm=None, lam=None):
    if lam is None:
        k = int((1 - p0 <= q) + (1 - p1 <= q))
    else:
        k = int(((1 - p0) / (1 + lam * vnorm) <= q) + ((1 - p1) / (1 + lam * vnorm) <= q))
    if k == 0:
        return "empty"
    if k == 1:
        return "single"
    return "two"


def comp_counts(labels):
    n = len(labels)
    e = sum(1 for s in labels if s == "empty")
    s = sum(1 for s in labels if s == "single")
    t = n - e - s
    return {
        "n": n,
        "empty": e,
        "single": s,
        "two": t,
        "pct_empty": e / n if n else np.nan,
        "pct_single": s / n if n else np.nan,
        "pct_two": t / n if n else np.nan,
    }

comp_v = [set_composition_row(p0[i], p1[i], qhat_van) for i in range(len(df))]
comp_a = [set_composition_row(p0[i], p1[i], qhat_linear, Vn[i], lam_linear) for i in range(len(df))]

setcomp_van = comp_counts(comp_v)
setcomp_linear = comp_counts(comp_a)

# Per-class coverage
cov_class = {}
for c in [0, 1]:
    idx = np.where(y == c)[0]
    if len(idx) > 0:
        cov_class[c] = {
            "vanilla": float(np.mean(scores_van[idx] <= qhat_van)),
            "linear": float(np.mean(scores_linear[idx] <= qhat_linear)),
            "n": int(len(idx)),
        }
    else:
        cov_class[c] = {"vanilla": np.nan, "linear": np.nan, "n": 0}

# Save augmented CSV + summary

df_out = df.copy()
df_out["score_van"] = scores_van
df_out["score_linear"] = scores_linear
df_out["covered_van"] = covered_van.astype(int)
df_out["covered_linear"] = covered_linear.astype(int)
df_out["Vn_center"] = Vn
df_out["set_van"] = comp_v
df_out["set_linear"] = comp_a
csv_aug = os.path.join(OUT_DIR, "points_with_cp_fields.csv")
df_out.to_csv(csv_aug, index=False)

summary = {
    "alpha": ALPHA,
    "T": T,
    "qhat": {"vanilla": qhat_van, "linear": qhat_linear},
    "lambda_linear": lam_linear,
    "variance_norm": {"min": VAR_MIN, "max": VAR_MAX},
    "coverage": {"vanilla": coverage_van, "linear": coverage_linear},
    "set_composition": {"vanilla": setcomp_van, "linear": setcomp_linear},
    "per_class_coverage": cov_class,
    "counts": {"N_points": int(len(df))},
    "files": {
        "points_with_scores_csv": Path(csv_points).name,
        "points_with_cp_fields_csv": Path(csv_aug).name,
    },
}
with open(os.path.join(OUT_DIR, "cp_eval_summary_2022_points.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("\nCP coverage on 2022 points")
print(f"  Vanilla : coverage={coverage_van:.3f} | sets: {setcomp_van}")
print(f"  Linear : coverage={coverage_linear:.3f} | lambda={lam_linear:.2f} | sets: {setcomp_linear}")
print("  Per-class coverage (n, cov_van, cov_linear):")
for c in [0, 1]:
    print(f"    class {c}: n={cov_class[c]['n']},  van={cov_class[c]['vanilla']:.3f},  lin={cov_class[c]['linear']:.3f}")

print(f"\nDone. Outputs saved to: {OUT_DIR}")
