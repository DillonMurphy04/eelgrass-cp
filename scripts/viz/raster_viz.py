"""
Raster visualizations for 2022 points: vanilla vs linear (lambda=3) vs nonparametric.
Requires the sensitivity cache and 2021 calibration meta; writes figures to OUT_DIR/viz_raster.
"""

import os, json
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show as rioshow
import fiona
from shapely.geometry import shape
from pathlib import Path
from src.eelgrass_cp import qhat_conformal

# -------- Config (match sensitivity_analysis.py) --------
RASTER_2022 = r"D:\\2022 Data\\Morro Bay Eelgrass AI - Rasterized Imagery\\2022_raster_final.tif"
POINTS_SHP  = r"D:\\2022 Data\\2022points\\2022points.shp"
OUT_DIR     = r"D:\\2022 Data\\cp_sensitivity_main"
CACHE_2022_NPZ = os.path.join(OUT_DIR, "points_2022_cache_for_sensitivity.npz")
META_JSON      = r"D:\\4 year training data\\2021\\ensemble_2020_2021_pixel_calib_cache\\pixel_calib_meta_2021.json"
CACHE_2021_NPZ = r"D:\\4 year training data\\2021\\ensemble_2020_2021_pixel_calib_cache\\pixel_calib_cache_2021.npz"

ALPHA   = 0.10           # should match meta["alpha"]
METHODS = ["vanilla", "linear", "nonparametric"]
FIG_DPI = 220

# -------- Helpers --------

def normalize_var(v, vmin, vmax):
    return np.clip((v - vmin) / max(vmax - vmin, 1e-8), 0.0, 1.0)


def compute_sigma_fn_and_qhats_from_2021(meta_json_path, cache_2021_npz,
                                         lam_linear=3.0, bins=20):
    """
    Reconstruct sigma(Vn) and q for:
      -  Nonparametric (nonparametric): s' = s / sigma(Vn)
      -  Linear (lambda=lam_linear):          s' = s / (1 + lambda Vn)

    using ONLY the 2021 CP cache + meta:
      -  Softmax w/ T on CP pool (idx_cp)
      -  Recreate cal/test split (CAL_TEST_SPLIT)
      -  Recreate cal_sub (CAL_TUNE_FRAC)
      -  Quantile-binned sigma(Vn) fit on cal subset
      -  q = (1-alpha) quantile of transformed scores on cal_sub
    """
    import pandas as pd
    from scipy.interpolate import interp1d
    from sklearn.model_selection import train_test_split

    D = np.load(cache_2021_npz, allow_pickle=True)
    with open(meta_json_path, "r") as f:
        meta = json.load(f)

    ALPHA_meta = float(meta["alpha"])
    T      = float(meta["temperature_T"])
    V_MIN  = float(meta["variance_min"])
    V_MAX  = float(meta["variance_max"])
    SEED   = int(meta.get("seed", 42))
    CAL_TEST_SPLIT = float(meta["splits"]["CAL_TEST_SPLIT"])
    CAL_TUNE_FRAC  = float(meta["splits"]["CAL_TUNE_FRAC"])

    # Softmax w/ T and CP pool
    def softmax_rows(logit_rows, T):
        z = logit_rows / max(T, 1e-8)
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return (e / np.clip(e.sum(axis=1, keepdims=True), 1e-12, None)).astype(np.float32)

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

    # Recreate cal/test and cal_sub splits
    idx_all = np.arange(N_pool)
    cal_idx, test_idx = train_test_split(
        idx_all, test_size=CAL_TEST_SPLIT, random_state=SEED, shuffle=True
    )
    cal_sub_idx, _ = train_test_split(
        cal_idx, test_size=CAL_TUNE_FRAC, random_state=SEED, shuffle=True
    )

    # sigma(Vn) via quantile bins on cal_idx
    import pandas as pd
    df = pd.DataFrame({"var": Vn_cp[cal_idx], "score": s_cp[cal_idx]})
    df["var_bin"] = pd.qcut(df["var"], q=bins, duplicates="drop")
    bin_means = df.groupby("var_bin", observed=False)["score"].mean().reset_index()
    bin_centers = np.array([iv.mid for iv in bin_means["var_bin"]], dtype=float)
    sigma_means = bin_means["score"].values.astype(float)

    sigma_fn = interp1d(
        bin_centers, sigma_means, kind="linear",
        fill_value="extrapolate", assume_sorted=False
    )

    # Mask for cal_sub in cal_idx index space
    pos_in_cal = {g: i for i, g in enumerate(cal_idx)}
    mask_cal = np.zeros(len(cal_idx), dtype=bool)
    for g in cal_sub_idx:
        if g in pos_in_cal:
            mask_cal[pos_in_cal[g]] = True

    # Nonparametric q: s' = s / sigma(Vn)
    s_cal  = s_cp[cal_idx]
    vn_cal = Vn_cp[cal_idx]
    s_nonparam_cal = s_cal / sigma_fn(vn_cal)
    qhat_nonparam = qhat_conformal(s_nonparam_cal[mask_cal], ALPHA_meta)

    # Linear (lambda=lam_linear) q fallback: s' = s / (1 + lambda Vn)
    s_linear_cal = s_cal / (1.0 + lam_linear * vn_cal)
    qhat_linear_fallback = qhat_conformal(s_linear_cal[mask_cal], ALPHA_meta)

    return sigma_fn, (V_MIN, V_MAX), qhat_nonparam, qhat_linear_fallback


def pretty_title(tag):
    if tag == "vanilla":
        return "Vanilla"
    if tag == "linear":
        return "Linear (lambda=3)"
    if tag == "nonparametric":
        return "Nonparametric"
    return tag


# -------- Load per-point cache and meta --------
npz = np.load(CACHE_2022_NPZ)
p0 = npz["p0"]; p1 = npz["p1"]; Vc = npz["V_center"]; y = npz["y"]; ids = npz["ids"]

with open(META_JSON, "r") as f:
    meta = json.load(f)

V_MIN = float(meta["variance_min"]); V_MAX = float(meta["variance_max"])
Vn = normalize_var(Vc, V_MIN, V_MAX)

# For sigma(V) and q from 2021
sigma_fn, _, qhat_nonparam_2021, qhat_linear_fallback = compute_sigma_fn_and_qhats_from_2021(
    META_JSON,
    CACHE_2021_NPZ,
    lam_linear=3.0,
    bins=20
)

# Base scores for 2022
py = np.where(y == 1, p1, p0)
s  = 1.0 - py

# --- q for each method (never from 2022) ---

# Vanilla: q from meta
if "pixel_level" not in meta or "vanilla" not in meta["pixel_level"]:
    raise RuntimeError("Vanilla q not found in meta JSON.")
qh_van = float(meta["pixel_level"]["vanilla"]["qhat"])

# Linear lambda=3 (tag still "linear" internally)
qh_linear = None
adaptive = meta.get("pixel_level", {}).get("adaptive", {})
linear_pack = adaptive.get("linear", adaptive.get("normalized", {}))
if linear_pack:
    for lam_str, pack in linear_pack.items():
        if abs(float(lam_str) - 3.0) < 1e-8:
            qh_linear = float(pack["qhat"])
            break
if qh_linear is None:
    qh_linear = qhat_linear_fallback  # fallback from 2021

# Nonparametric: always from 2021
qh_nonparam = qhat_nonparam_2021

# Define transforms (lambda=3 now)
s_linear  = s / (1 + 3.0 * Vn)      # Linear lambda=3
s_nonparam  = s / sigma_fn(Vn)        # Nonparametric normalizer

# For sets, reuse transformed scores for each class-wise score
trans_linear = lambda sclass: sclass / (1 + 3.0 * Vn)
trans_nonparam  = lambda sclass: sclass / sigma_fn(Vn)

# Set membership per method
p0s, p1s = 1 - p0, 1 - p1

cats = {}
for tag, qh, trans in [
    ("vanilla",         qh_van,  None),
    ("linear",   qh_linear, trans_linear),   # Linear lambda=3
    ("nonparametric", qh_nonparam,  trans_nonparam),    # Nonparametric
]:
    s0 = p0s if trans is None else trans(p0s)
    s1 = p1s if trans is None else trans(p1s)
    k  = (s0 <= qh).astype(int) + (s1 <= qh).astype(int)
    # Categories: 0=empty, 1=single, 2=two
    pred_single = (k == 1)
    pred_two    = (k == 2)
    pred_empty  = (k == 0)
    # Predicted class under single-set
    pred_cls = np.where((s0 <= qh) & ~(s1 <= qh), 0,
                np.where((s1 <= qh) & ~(s0 <= qh), 1, -1))
    correct = (pred_cls == y)
    cats[tag] = {
        "empty": pred_empty,
        "single_correct": pred_single & correct,
        "single_wrong": pred_single & (~correct),
        "two": pred_two,
    }

# -------- Plotting --------
from rasterio.windows import from_bounds
from rasterio.enums import Resampling

figs_dir = os.path.join(OUT_DIR, "viz_raster")
os.makedirs(figs_dir, exist_ok=True)

# Pull shapefile point coords (map space)
with fiona.open(POINTS_SHP, 'r') as src:
    xs, ys, gts = [], [], []
    for feat in src:
        geom = shape(feat['geometry'])
        xs.append(geom.x); ys.append(geom.y)
        gts.append(int(feat['properties'].get('GROUND_TRU', 0)))
xs = np.asarray(xs); ys = np.asarray(ys); gts = np.asarray(gts)

# Make a few useful masks from computed categories
def setsize_from_cats(c):
    # 0 empty, 1 single (correct OR wrong), 2 two
    return (c["two"].astype(int) * 2 +
            (c["single_correct"] | c["single_wrong"]).astype(int))

setsize_van = setsize_from_cats(cats["vanilla"])
covered_van = (setsize_van > 0)

if "linear" in cats:
    setsize_linear = setsize_from_cats(cats["linear"])
    covered_linear = (setsize_linear > 0)
    expanded_linear = covered_linear & (~covered_van)
else:
    expanded_linear = np.zeros_like(covered_van, dtype=bool)

# Optional "focus" logic to avoid massive windows
FOCUS = "expanded"   # "expanded" | "high_var" | "all"
MARGIN = 50.0        # map units around focus bbox
TARGET_LONG_SIDE = 2000  # downsample longer side to this many pixels

if FOCUS == "expanded" and expanded_linear.any():
    focus_mask = expanded_linear
elif FOCUS == "high_var":
    qv = np.quantile(Vn, 0.90)
    focus_mask = (Vn >= qv)
else:
    focus_mask = np.ones_like(xs, dtype=bool)

fx, fy = xs[focus_mask], ys[focus_mask]
if len(fx) == 0:  # fallback
    fx, fy = xs, ys

xmin, xmax = fx.min() - MARGIN, fx.max() + MARGIN
ymin, ymax = fy.min() - MARGIN, fy.max() + MARGIN

# Helper to read a cropped + downsampled RGB + extent once
def read_crop_rgb_with_extent(ds, xmin, ymin, xmax, ymax, target_long_side):
    # clamp to dataset bounds
    xmin = max(xmin, ds.bounds.left);  xmax = min(xmax, ds.bounds.right)
    ymin = max(ymin, ds.bounds.bottom); ymax = min(ymax, ds.bounds.top)
    win = from_bounds(xmin, ymin, xmax, ymax, ds.transform).round_offsets().round_lengths()
    h, w = int(win.height), int(win.width)
    scale = max(h, w) / float(target_long_side) if max(h, w) > target_long_side else 1.0
    out_h, out_w = max(1, int(h/scale)), max(1, int(w/scale))
    # read (C,H,W) and normalize per crop for display
    bands = [1,2,3] if ds.count >= 3 else [1,1,1]
    rgb = ds.read(bands, window=win, out_shape=(len(bands), out_h, out_w),
                  resampling=Resampling.bilinear).astype(np.float32)
    mn, mx = rgb.min(), rgb.max()
    if mx > mn:
        rgb = (rgb - mn) / (mx - mn + 1e-12)
    # compute map extent for imshow
    win_transform = ds.window_transform(win)
    # scale transform to the downsampled grid
    win_transform = win_transform * win_transform.scale(win.width/float(out_w), win.height/float(out_h))
    x0, y0 = win_transform * (0, 0)
    x1, y1 = win_transform * (out_w, out_h)
    extent = (x0, x1, y1, y0)  # origin='upper'
    return rgb, extent, (xmin, ymin, xmax, ymax)

# Read the crop ONCE
with rasterio.open(RASTER_2022) as ds:
    rgb, extent, (xmin, ymin, xmax, ymax) = read_crop_rgb_with_extent(
        ds, xmin, ymin, xmax, ymax, TARGET_LONG_SIDE
    )

# Subset points to crop bbox
inside = (xs >= xmin) & (xs <= xmax) & (ys >= ymin) & (ys <= ymax)
X = xs[inside]; Y = ys[inside]; GT = gts[inside]

def scatter_categories(ax, tag, title):
    """Draw full categories for a method on the cropped window."""
    ax.imshow(np.transpose(rgb, (1,2,0)), extent=extent, origin="upper")
    ax.set_title(title, pad=2, fontsize=9)
    ax.set_aspect('equal', adjustable='box')
    ax.axis("off")

    c = cats[tag]
    # subset masks to the crop
    m_empty   = c["empty"][inside]
    m_two     = c["two"][inside]
    m_sc      = c["single_correct"][inside]
    m_sw      = c["single_wrong"][inside]

    # draw singles first (under)
    ax.scatter(X[m_sc], Y[m_sc], s=10, c="#20a320", alpha=0.75, lw=0,
               label="Singleton (correct)", zorder=2)
    ax.scatter(X[m_sw], Y[m_sw], s=10, c="#cc3737", alpha=0.75, lw=0,
               label="Singleton (incorrect)", zorder=2)
    # rare types on top
    ax.scatter(X[m_empty], Y[m_empty], marker='x', s=18, c="black",
               alpha=0.9, lw=1.2, label="Empty set", zorder=3)
    ax.scatter(X[m_two],   Y[m_two],   marker='s', s=18, facecolors='none',
               edgecolors="black", alpha=0.95, lw=1.3, label="Two-label set", zorder=3)


# Build categorical labels for change detection: -1=none, 0=empty, 1=single_correct, 2=single_wrong, 3=two
label_map = {}
for tag, c in cats.items():
    lab = np.full(len(xs), -1, dtype=np.int8)
    lab[c["empty"]]          = 0
    lab[c["single_correct"]] = 1
    lab[c["single_wrong"]]   = 2
    lab[c["two"]]            = 3
    label_map[tag] = lab

def scatter_changes(ax, to_tag, title):
    """
    Draw only points whose category changed between vanilla and to_tag,
    colored/marked by their NEW category under to_tag.
    """
    ax.imshow(np.transpose(rgb, (1,2,0)), extent=extent, origin="upper")
    ax.set_title(title, pad=2, fontsize=9)
    ax.set_aspect('equal', adjustable='box')
    ax.axis("off")

    labels_from = label_map["vanilla"]
    labels_to   = label_map[to_tag]
    changed = inside & (labels_from != labels_to)

    c_to = cats[to_tag]
    # masks for new category among changed points
    m_empty = c_to["empty"] & changed
    m_two   = c_to["two"] & changed
    m_sc    = c_to["single_correct"] & changed
    m_sw    = c_to["single_wrong"] & changed

    # Slightly larger markers so changes pop
    ax.scatter(xs[m_sc],   ys[m_sc],   s=20, c="#20a320", alpha=0.85, lw=0,
               label="Singleton (correct)", zorder=3)
    ax.scatter(xs[m_sw],   ys[m_sw],   s=20, c="#cc3737", alpha=0.85, lw=0,
               label="Singleton (incorrect)", zorder=3)
    ax.scatter(xs[m_empty], ys[m_empty], marker='x', s=26, c="black",
               alpha=0.9, lw=1.4, label="Empty set", zorder=4)
    ax.scatter(xs[m_two],   ys[m_two],   marker='s', s=26, facecolors='none',
               edgecolors="black", alpha=0.95, lw=1.6, label="Two-label set", zorder=4)


# ========================= Figures (A, B) =========================

# (A) Base raster only
plt.figure(figsize=(5.0, 3.5))
plt.imshow(np.transpose(rgb, (1,2,0)), extent=extent, origin="upper")
plt.axis('off')
plt.title("2022 raster (RGB)", pad=2, fontsize=9)
plt.tight_layout(pad=0.05)
plt.savefig(os.path.join(figs_dir, "A_raster_only_CROP.png"),
            dpi=FIG_DPI, bbox_inches="tight")
plt.close()

# (B) Raster + ground truth points
plt.figure(figsize=(5.0, 3.5))
plt.imshow(np.transpose(rgb, (1,2,0)), extent=extent, origin="upper")
m0, m1 = (GT == 0), (GT == 1)
plt.scatter(X[m0], Y[m0], s=8, c="#7b68ee", alpha=0.7, label="GT=0", zorder=2)
plt.scatter(X[m1], Y[m1], s=8, c="#2ca02c", alpha=0.7, label="GT=1", zorder=2)
plt.title("Ground-truth points", pad=2, fontsize=9)
plt.axis('off')
plt.legend(loc="lower right", fontsize=7, frameon=True)
plt.tight_layout(pad=0.05)
plt.savefig(os.path.join(figs_dir, "B_raster_gt_CROP.png"),
            dpi=FIG_DPI, bbox_inches="tight")
plt.close()

# ========================= Figures (C, D) - multi-panel, tight & large =========================

# (C) Triptych: Vanilla / Linear(lambda=3) / Nonparametric with shared legend
methods_panel = [m for m in ["vanilla", "linear", "nonparametric"] if m in cats]
if methods_panel:
    n = len(methods_panel)
    # Tall, relatively narrow figure so panels fill almost everything
    fig, axes = plt.subplots(1, n, figsize=(7.5, 6.0), sharex=True, sharey=True)
    if n == 1:
        axes = [axes]

    for ax, meth in zip(axes, methods_panel):
        title = f"{pretty_title(meth)} (alpha={ALPHA})"
        scatter_categories(ax, meth, title)

    # Shared legend below
    handles, labels = axes[-1].get_legend_handles_labels()

    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.07),
        frameon=True,
        fontsize=7,
    )

    fig.subplots_adjust(
        left=0.01,
        right=0.99,
        top=0.88,
        bottom=0.12,
        wspace=0.02
    )

    plt.savefig(
        os.path.join(figs_dir, f"C_all_methods_diag_a{ALPHA}_CROP.png"),
        dpi=FIG_DPI,
        bbox_inches="tight"   # crop outer white margins
    )
    plt.close()

# (D) Changes-only panel: Vanilla -> Linear(lambda=3) and Vanilla -> Nonparametric
methods_change = [m for m in ["linear", "nonparametric"] if m in cats]
if methods_change:
    n = len(methods_change)
    fig, axes = plt.subplots(1, n, figsize=(7.5, 6.0), sharex=True, sharey=True)
    if n == 1:
        axes = [axes]

    for ax, meth in zip(axes, methods_change):
        title = f"Changes: Vanilla -> {pretty_title(meth)}"
        scatter_changes(ax, meth, title)

    handles, labels = axes[-1].get_legend_handles_labels()

    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.07),
        frameon=True,
        fontsize=7,
    )

    # Reduce bottom margin so panels sit closer to the legend
    fig.subplots_adjust(
        left=0.01,
        right=0.99,
        top=0.88,
        bottom=0.12,
        wspace=0.02
    )

    plt.savefig(
        os.path.join(figs_dir, f"D_changes_from_vanilla_a{ALPHA}_CROP.png"),
        dpi=FIG_DPI,
        bbox_inches="tight"
    )
    plt.close()


print(" Raster visuals exported (cropped/windowed):", figs_dir)
