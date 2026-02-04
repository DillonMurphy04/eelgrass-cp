"""
Single-chip visualization: RGB + set overlays for vanilla, linear (lambda=3), and nonparametric.
Outputs a 2x2 panel figure and optional center-pixel diagnostics.
"""

import os, json, importlib, warnings
import numpy as np
import rasterio
from rasterio.windows import Window
import torch
import fiona
from shapely.geometry import shape
import matplotlib.pyplot as plt
from eelgrass_cp import qhat_conformal

# -------------------------------------------------------
# User Options
# -------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Choose which point index to visualize (0-based in shapefile order)
POINT_INDEX_TO_VISUALIZE = 0

# Save location/name
OUT_NAME = "chip_set_compare_point_{:04d}.png"
FIG_DPI  = 260

# Data/paths (from  chip script)
RASTER_PATH = r"D:\2022 Data\Morro Bay Eelgrass AI - Rasterized Imagery\2022_raster_final.tif"
POINTS_SHP  = r"D:\2022 Data\2022points\2022points.shp"
OUT_DIR     = r"D:\2022 Data\point_chips_validation_ensemble"
FIG_DIR     = os.path.join(OUT_DIR, "figs_chip_set_compare")
os.makedirs(FIG_DIR, exist_ok=True)

# Chip & label config
CHIP_SIZE    = 448
GT_FIELD     = "GROUND_TRU"
TARGET_CLASS = 1   # eelgrass class index for prob/var

# EMD model paths
MODEL_EMDS = {
    "unet_4yr":   r"D:\4yr_Unet_model\models\checkpoint_2025-09-19_22-47-55_epoch_11\checkpoint_2025-09-19_22-47-55_epoch_11.emd",
    "samlora_4yr": r"D:\sam_lora_model\samlora_4_year\models\checkpoint_2025-06-11_18-24-03_epoch_18\checkpoint_2025-06-11_18-24-03_epoch_18.emd",
    "samlora_2021": r"D:\sam_lora_model_2021\models\checkpoint_2025-07-30_18-52-20_epoch_18\checkpoint_2025-07-30_18-52-20_epoch_18.emd",
    "deeplab_2021": r"D:\DeepLab_2021\DeepLab_2021.emd",
    "unet_2021":    r"D:\2021 Unet\checkpoint_2025-06-25_12-39-26_epoch_11.emd",
    "deeplab_4yr":  r"D:\deeplab_4_year\deeplab_4_year.emd"
}

# 2021 calibration artifacts (same as raster_viz.py)
META_JSON      = r"D:\4 year training data\2021\ensemble_2020_2021_pixel_calib_cache\pixel_calib_meta_2021.json"
CACHE_2021_NPZ = r"D:\4 year training data\2021\ensemble_2020_2021_pixel_calib_cache\pixel_calib_cache_2021.npz"

# CP / normalization params
ALPHA = 0.10
LAM   = 3.0
BINS  = 20

# Overlay appearance
OVERLAY_ALPHA = 0.55


# -------------------------------------------------------
# Utilities (from  chip script)
# -------------------------------------------------------
def load_model_and_config(emd_path):
    """Load arcgis.learn model from EMD and extract normalization config."""
    with open(emd_path, "r") as f:
        data = json.load(f)

    model_name = data["ModelName"]
    ModelClass = getattr(importlib.import_module("arcgis.learn"), model_name)

    model_obj = ModelClass.from_model(data=None, emd_path=emd_path)
    model = model_obj.learn.model.to(DEVICE)
    model.eval()

    if "NormalizationStats" in data:
        norm = data["NormalizationStats"]
        config = {
            "chip_height": data.get("ImageHeight", CHIP_SIZE),
            "chip_width": data.get("ImageWidth", CHIP_SIZE),
            "min": np.array(norm["band_min_values"], dtype=np.float32),
            "max": np.array(norm["band_max_values"], dtype=np.float32),
            "scaled_mean": np.array(norm["scaled_mean_values"], dtype=np.float32),
            "scaled_std": np.array(norm["scaled_std_values"], dtype=np.float32),
        }
    else:
        config = {
            "chip_height": data.get("ImageHeight", CHIP_SIZE),
            "chip_width": data.get("ImageWidth", CHIP_SIZE),
            "min": np.array([0.0, 0.0, 0.0], dtype=np.float32),
            "max": np.array([255.0, 255.0, 255.0], dtype=np.float32),
            "scaled_mean": np.array([0.485, 0.456, 0.406], dtype=np.float32),
            "scaled_std": np.array([0.229, 0.224, 0.225], dtype=np.float32),
        }

    return model, config


def norm_chip(chip, config):
    """Per-EMD normalization: min-max to [0,1], then standardize with scaled_mean/std."""
    chip_hw_c = np.transpose(chip, (1, 2, 0)).astype(np.float32)  # (H, W, C)
    denom = (config["max"] - config["min"])
    denom = np.where(denom == 0, 1.0, denom)
    scaled = (chip_hw_c - config["min"]) / denom
    scaled = (scaled - config["scaled_mean"]) / np.where(config["scaled_std"] == 0, 1.0, config["scaled_std"])
    return np.transpose(scaled, (2, 0, 1))  # (C, H, W)


@torch.no_grad()
def model_probs(model, chip_norm):
    """Forward pass -> softmax probabilities. Returns numpy float32 (C, H, W)."""
    x = torch.from_numpy(chip_norm).float().unsqueeze(0).to(DEVICE)  # (1, C, H, W)
    out = model(x)
    if isinstance(out, (list, tuple)):
        out = out[0]
    probs = torch.nn.functional.softmax(out, dim=1)
    return probs.squeeze(0).detach().cpu().numpy().astype(np.float32)  # (C, H, W)


def safe_read_chip(ds, row, col, size):
    half = size // 2
    window = Window(col - half, row - half, size, size)
    chip = ds.read(window=window)  # (C, H, W)
    if chip.shape[1] < size or chip.shape[2] < size:
        return None
    return chip


def ensure_uint8_rgb(chip):
    """Return an RGB uint8 array (H, W, 3) for visualization."""
    c = min(3, chip.shape[0])
    rgb = np.moveaxis(chip[:c], 0, -1)
    arr = rgb.astype(np.float32)
    vmin = np.percentile(arr, 1)
    vmax = np.percentile(arr, 99)
    if vmax <= vmin:
        vmax = vmin + 1.0
    arr = (np.clip((arr - vmin) / (vmax - vmin), 0, 1) * 255.0).astype(np.uint8)
    if arr.shape[2] < 3:
        arr = np.repeat(arr, 3, axis=2)
    return arr


# -------------------------------------------------------
# CP / set overlay helpers
# -------------------------------------------------------
def normalize_var(v, vmin, vmax):
    return np.clip((v - vmin) / max(vmax - vmin, 1e-8), 0.0, 1.0)


def softmax_rows(logit_rows, T):
    z = logit_rows / max(T, 1e-8)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return (e / np.clip(e.sum(axis=1, keepdims=True), 1e-12, None)).astype(np.float32)


def compute_sigma_fn_and_qhats_from_2021(meta_json_path, cache_2021_npz, lam_linear=3.0, bins=20):
    """
    Reconstruct sigma_hat(Vn) and qhats using ONLY the 2021 CP cache + meta.
    Matches  raster_viz.py.
    """
    import pandas as pd
    from scipy.interpolate import interp1d
    from sklearn.model_selection import train_test_split

    D = np.load(cache_2021_npz, allow_pickle=True)
    with open(meta_json_path, "r") as f:
        meta = json.load(f)

    alpha_meta = float(meta["alpha"])
    T      = float(meta["temperature_T"])
    V_MIN  = float(meta["variance_min"])
    V_MAX  = float(meta["variance_max"])
    SEED   = int(meta.get("seed", 42))
    CAL_TEST_SPLIT = float(meta["splits"]["CAL_TEST_SPLIT"])
    CAL_TUNE_FRAC  = float(meta["splits"]["CAL_TUNE_FRAC"])

    PLOG_all = D["PLOG"]
    Y_all    = D["Y"]
    V_all    = D["V"]
    idx_cp   = D.get("idx_cp", None)
    pool = np.arange(len(Y_all)) if idx_cp is None else idx_cp

    P_cp = softmax_rows(PLOG_all[pool], T=T)
    Y_cp = Y_all[pool]
    V_cp = V_all[pool]

    N_pool = len(P_cp)
    py_cp = P_cp[np.arange(N_pool), Y_cp]
    s_cp  = 1.0 - py_cp
    Vn_cp = normalize_var(V_cp, V_MIN, V_MAX)

    idx_all = np.arange(N_pool)
    cal_idx, _ = train_test_split(
        idx_all, test_size=CAL_TEST_SPLIT, random_state=SEED, shuffle=True
    )
    cal_sub_idx, _ = train_test_split(
        cal_idx, test_size=CAL_TUNE_FRAC, random_state=SEED, shuffle=True
    )
    cal_sub_set = set(cal_sub_idx.tolist())
    mask_cal_sub = np.array([i in cal_sub_set for i in cal_idx], dtype=bool)

    df = pd.DataFrame({"var": Vn_cp[cal_idx], "score": s_cp[cal_idx]})
    df["var_bin"] = pd.qcut(df["var"], q=bins, duplicates="drop")
    bin_means = df.groupby("var_bin", observed=False)["score"].mean().reset_index()
    bin_centers = np.array([iv.mid for iv in bin_means["var_bin"]], dtype=float)
    sigma_means = bin_means["score"].values.astype(float)

    sigma_fn = interp1d(
        bin_centers, sigma_means, kind="linear",
        fill_value="extrapolate", assume_sorted=False
    )

    s_cal  = s_cp[cal_idx]
    vn_cal = Vn_cp[cal_idx]

    s_nonparam_cal = s_cal / sigma_fn(vn_cal)
    qhat_nonparam = qhat_conformal(s_nonparam_cal[mask_cal_sub], alpha_meta)

    s_linear_cal = s_cal / (1.0 + lam_linear * vn_cal)
    qhat_linear = qhat_conformal(s_linear_cal[mask_cal_sub], alpha_meta)

    return sigma_fn, (V_MIN, V_MAX), qhat_nonparam, qhat_linear


def make_label_map(k, singleton_class):
    """
    4-way label image for overlay:
      1 = singleton {other}     (class 0 only)
      2 = singleton {eelgrass}  (class 1 only)
      3 = two-label set
      4 = empty set
    """
    lab = np.zeros_like(k, dtype=np.uint8)
    lab[(k == 1) & (singleton_class == 0)] = 1
    lab[(k == 1) & (singleton_class == 1)] = 2
    lab[k == 2] = 3
    lab[k == 0] = 4
    return lab


def overlay_discrete(ax, rgb_uint8, lab, center_xy=None):
    """Overlay discrete categories with fixed palette."""
    ax.imshow(rgb_uint8)
    ax.axis("off")

    H, W, _ = rgb_uint8.shape
    overlay = np.zeros((H, W, 4), dtype=np.float32)

    # singleton-other (light green), singleton-eelgrass (dark green),
    # two-label (yellow), empty (dark gray)
    overlay[lab == 1] = (0.55, 0.85, 0.45, OVERLAY_ALPHA)  # singleton {other}
    overlay[lab == 2] = (0.12, 0.70, 0.12, OVERLAY_ALPHA)  # singleton {eelgrass}
    overlay[lab == 3] = (0.95, 0.80, 0.10, OVERLAY_ALPHA)  # two-label
    overlay[lab == 4] = (0.20, 0.20, 0.20, OVERLAY_ALPHA)  # empty
    ax.imshow(overlay)

    if center_xy is not None:
        cy, cx = center_xy
        ax.plot(cx, cy, "wo", markersize=4)


def build_legend():
    from matplotlib.patches import Patch
    return [
        Patch(facecolor=(0.55, 0.85, 0.45, OVERLAY_ALPHA), edgecolor="none", label="Singleton {other}"),
        Patch(facecolor=(0.12, 0.70, 0.12, OVERLAY_ALPHA), edgecolor="none", label="Singleton {eelgrass}"),
        Patch(facecolor=(0.95, 0.80, 0.10, OVERLAY_ALPHA), edgecolor="none", label="Two-label set"),
        Patch(facecolor=(0.20, 0.20, 0.20, OVERLAY_ALPHA), edgecolor="none", label="Empty set"),
    ]


def pretty_title(tag):
    if tag == "raw":
        return "Raw chip (RGB)"
    if tag == "vanilla":
        return f"Vanilla (alpha={ALPHA})"
    if tag == "linear":
        return f"Linear (lambda={LAM}) (alpha={ALPHA})"
    if tag == "nonparam":
        return f"Nonparametric (alpha={ALPHA})"
    return tag


def describe_center_set(inc0, inc1, cy, cx):
    """Human-readable center-pixel set description."""
    i0 = bool(inc0[cy, cx])
    i1 = bool(inc1[cy, cx])
    if i0 and i1:
        return "two-label {other, eelgrass}"
    if i1 and (not i0):
        return "singleton {eelgrass}"
    if i0 and (not i1):
        return "singleton {other}"
    return "empty set"


# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main():
    # ---------------- Load models ----------------
    print(" Loading models...")
    ensemble = {}
    for key, emd in MODEL_EMDS.items():
        try:
            model, config = load_model_and_config(emd)
            ensemble[key] = {"model": model, "config": config}
            print(f"  -  Loaded {key}")
        except Exception as e:
            warnings.warn(f"Failed to load {key} ({emd}): {e}")

    if len(ensemble) == 0:
        raise RuntimeError("No models loaded; please check EMD paths.")

    # ---------------- Read points ----------------
    print(" Reading points...")
    with fiona.open(POINTS_SHP, "r") as src:
        points = [(shape(feat["geometry"]), feat["properties"]) for feat in src]
    print(f"  -  Total points: {len(points)}")

    if POINT_INDEX_TO_VISUALIZE < 0 or POINT_INDEX_TO_VISUALIZE >= len(points):
        raise ValueError(f"POINT_INDEX_TO_VISUALIZE={POINT_INDEX_TO_VISUALIZE} out of range.")

    geom, props = points[POINT_INDEX_TO_VISUALIZE]
    x_map, y_map = geom.x, geom.y

    gt_val = props.get(GT_FIELD, None)
    gt_int = int(gt_val) if gt_val is not None else None

    # ---------------- Read chip + run ensemble ----------------
    print(f" Reading chip + running ensemble for point {POINT_INDEX_TO_VISUALIZE}...")
    with rasterio.open(RASTER_PATH) as ds:
        row, col = ds.index(x_map, y_map)
        chip = safe_read_chip(ds, row, col, CHIP_SIZE)
        if chip is None:
            raise RuntimeError("Chip read failed (point likely near edge).")

    per_model_probmaps = []
    num_classes = None

    for key, pack in ensemble.items():
        try:
            chip_norm = norm_chip(chip, pack["config"])
            probs = model_probs(pack["model"], chip_norm)  # (C,H,W)

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
        except Exception as e:
            warnings.warn(f"Inference failed for model {key}: {e}")

    if len(per_model_probmaps) == 0:
        raise RuntimeError("No model predictions were produced for this chip.")

    # Stack: (M, C, H, W)
    stack = np.stack(per_model_probmaps, axis=0)
    mean_probs = np.mean(stack, axis=0)  # (C,H,W)

    tc = min(TARGET_CLASS, mean_probs.shape[0] - 1)
    var_map = np.var(stack[:, tc, :, :], axis=0).astype(np.float32)  # (H,W)
    hard_mask = np.argmax(mean_probs, axis=0).astype(np.int32)        # (H,W)

    # Center pixel
    cy, cx = CHIP_SIZE // 2, CHIP_SIZE // 2
    center_mean = float(mean_probs[tc, cy, cx])
    center_var  = float(var_map[cy, cx])
    center_pred = int(hard_mask[cy, cx])

    print(f"  -  Center stats: mean_prob={center_mean:.4f}, var={center_var:.6f}, hard_pred={center_pred}, gt={gt_int}")

    # ---------------- Load 2021 meta + compute qhats/sigma ----------------
    with open(META_JSON, "r") as f:
        meta = json.load(f)

    V_MIN = float(meta["variance_min"])
    V_MAX = float(meta["variance_max"])

    sigma_fn, _, qhat_nonparam_2021, qhat_linear_2021 = compute_sigma_fn_and_qhats_from_2021(
        META_JSON, CACHE_2021_NPZ, lam_linear=LAM, bins=BINS
    )

    if "pixel_level" not in meta or "vanilla" not in meta["pixel_level"]:
        raise RuntimeError("Vanilla qhat not found: meta['pixel_level']['vanilla']['qhat']")
    qh_van = float(meta["pixel_level"]["vanilla"]["qhat"])

    # Linear qhat: prefer meta value at lambda, else fallback
    qh_lin = None
    adaptive = meta.get("pixel_level", {}).get("adaptive", {})
    linear_pack = adaptive.get("linear", adaptive.get("normalized", {}))
    if linear_pack:
        for lam_str, pack in linear_pack.items():
            if abs(float(lam_str) - float(LAM)) < 1e-8:
                qh_lin = float(pack["qhat"])
                break
    if qh_lin is None:
        qh_lin = qhat_linear_2021

    qh_np = qhat_nonparam_2021

    # ---------------- Build set overlays (pixelwise) ----------------
    # Binary class probs (use ensemble mean prob-map for TARGET_CLASS)
    p1 = mean_probs[tc].astype(np.float32)  # eelgrass prob
    p0 = (1.0 - p1).astype(np.float32)

    # Class-wise scores: s_c = 1 - p_c
    s0 = 1.0 - p0
    s1 = 1.0 - p1

    # Vn per pixel (normalize chip variance using global 2021 min/max)
    Vn = normalize_var(var_map, V_MIN, V_MAX).astype(np.float32)

    # Transforms
    trans_lin = lambda s: s / (1.0 + float(LAM) * Vn)
    trans_np  = lambda s: s / sigma_fn(Vn)

    def infer_sets(qhat, trans=None):
        """
        Returns:
          k: (H,W) in {0,1,2} number of labels included
          singleton_class: (H,W) in {0,1,-1} where k==1 gives which singleton label, else -1
          inc0/inc1: bool masks for set membership (other/eelgrass)
        """
        if trans is None:
            t0, t1 = s0, s1
        else:
            t0, t1 = trans(s0), trans(s1)

        inc0 = (t0 <= qhat)  # other included
        inc1 = (t1 <= qhat)  # eelgrass included

        k = inc0.astype(np.uint8) + inc1.astype(np.uint8)

        singleton_class = np.full(k.shape, -1, dtype=np.int8)
        singleton_class[(inc0) & (~inc1)] = 0
        singleton_class[(inc1) & (~inc0)] = 1

        return k, singleton_class, inc0, inc1

    k_van, sc_van, inc0_van, inc1_van = infer_sets(qh_van, None)
    k_lin, sc_lin, inc0_lin, inc1_lin = infer_sets(qh_lin, trans_lin)
    k_np,  sc_np,  inc0_np,  inc1_np  = infer_sets(qh_np,  trans_np)

    # ---- Explicit check: what is the CENTER pixel set for each method? ----
    print("  -  Center-pixel conformal sets:")
    print(f"      Vanilla:         {describe_center_set(inc0_van, inc1_van, cy, cx)}")
    print(f"      Linear (lambda={LAM}): {describe_center_set(inc0_lin, inc1_lin, cy, cx)}")
    print(f"      Nonparametric:    {describe_center_set(inc0_np,  inc1_np,  cy, cx)}")

    lab_van = make_label_map(k_van, sc_van)
    lab_lin = make_label_map(k_lin, sc_lin)
    lab_np  = make_label_map(k_np,  sc_np)

    # ---------------- Plot 2x2 figure ----------------
    rgb_uint8 = ensure_uint8_rgb(chip)

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 9.0))
    for ax in axes.ravel():
        ax.axis("off")

    # Top-left: Raw
    axes[0, 0].imshow(rgb_uint8)
    axes[0, 0].plot(cx, cy, "ro", markersize=4)
    axes[0, 0].set_title(pretty_title("raw"), fontsize=11, pad=6)

    # Top-right: Vanilla
    overlay_discrete(axes[0, 1], rgb_uint8, lab_van, center_xy=(cy, cx))
    axes[0, 1].set_title(pretty_title("vanilla"), fontsize=11, pad=6)

    # Bottom-left: Linear
    overlay_discrete(axes[1, 0], rgb_uint8, lab_lin, center_xy=(cy, cx))
    axes[1, 0].set_title(pretty_title("linear"), fontsize=11, pad=6)

    # Bottom-right: Nonparametric
    overlay_discrete(axes[1, 1], rgb_uint8, lab_np, center_xy=(cy, cx))
    axes[1, 1].set_title(pretty_title("nonparam"), fontsize=11, pad=6)

    # Shared legend
    handles = build_legend()
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        frameon=True,
        fontsize=11,
        handlelength=2.0,
        handleheight=1.2,
        labelspacing=0.8,
        columnspacing=1.8,
        bbox_to_anchor=(0.5, 0.0)
    )

    fig.subplots_adjust(
        left=0.03,
        right=0.97,
        top=0.92,
        bottom=0.12,
        hspace=0.08,
        wspace=0.00
    )

    out_path = os.path.join(FIG_DIR, OUT_NAME.format(POINT_INDEX_TO_VISUALIZE))
    plt.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()

    print(" Exported:", out_path)


if __name__ == "__main__":
    main()
