"""
Three-panel cropped overlay: ground truth vs vanilla vs linear (lambda=3).
Outputs triptych and change-only figures to OUT_DIR.
"""
import os, json
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
import matplotlib.pyplot as plt

# ---- Inputs ----
RASTER_PATH   = r"D:\2022 Data\Morro Bay Eelgrass AI - Rasterized Imagery\2022_raster_final.tif"
CSV_POINTS    = r"D:\2022 Data\cp_spatial_blocks_2022\points_with_blocks.csv"   # has x,y, p0,p1,Vn, gt
META_JSON     = r"D:\4 year training data\2021\ensemble_2020_2021_pixel_calib_cache\pixel_calib_meta_2021.json"

OUT_DIR       = r"D:\2022 Data\cp_spatial_blocks_2022"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_FIG       = os.path.join(OUT_DIR, "overlay_triptych_GT_v_Vanilla_v_Linear_l3_CROPPED.png")

FOCUS = "expanded"   # "expanded" | "all" | "high_var" | ("block", k) e.g. set FOCUS=("block", 3)
MARGIN = 50.0        # map units around the focus bbox
TARGET_LONG_SIDE = 2000  # downsample longer side to this many pixels

# ---- Load meta (q-hats and variance min/max; Vn already provided here) ----
with open(META_JSON, "r") as f:
    meta = json.load(f)
qhat_vanilla = float(meta["pixel_level"]["vanilla"]["qhat"])

# CHANGED TO LAMBDA = 3
qhat_linear = None
adaptive = meta.get("pixel_level", {}).get("adaptive", {})
linear_pack = adaptive.get("linear", adaptive.get("normalized", {}))
for lam_str, pack in linear_pack.items():
    if abs(float(lam_str) - 3.0) < 1e-6:
        qhat_linear = float(pack["qhat"])
        break
if qhat_linear is None:
    raise RuntimeError("qhat for linear lambda=3 not found in META_JSON.")

# ---- Load points ----
df = pd.read_csv(CSV_POINTS)

required_cols = {"x","y","p0","p1","Vn","gt"}
missing = required_cols - set(df.columns)
if missing:
    raise RuntimeError(f"CSV missing columns: {missing}")

# If vanilla/linear covered flags are not present in the CSV, recompute below.

# ---- Helper: compute set membership (empty / single / two) and covered flags ----
def compute_sets(p0, p1, q, Vn=None, lam=None):
    """
    Returns:
      covered (0/1),
      set_size in {0,1,2},
      which_single (np.nan if not single else 0 or 1)
    """
    s0 = 1 - p0
    s1 = 1 - p1
    if Vn is not None and lam is not None:
        s0 = s0 / (1 + lam * Vn)
        s1 = s1 / (1 + lam * Vn)
    in0 = s0 <= q
    in1 = s1 <= q
    set_size = in0.astype(int) + in1.astype(int)
    covered = (set_size >= 1).astype(int)
    # which class if single
    which_single = np.full_like(set_size, fill_value=np.nan, dtype=float)
    single_mask = (set_size == 1)
    which_single[single_mask & in1] = 1.0
    which_single[single_mask & in0] = 0.0
    return covered, set_size, which_single

# Vanilla
cov_v, size_v, single_v = compute_sets(df["p0"].values, df["p1"].values, qhat_vanilla)
# Linear lambda=3
lam = 3.0
cov_linear, size_linear, single_linear = compute_sets(df["p0"].values, df["p1"].values, qhat_linear, Vn=df["Vn"].values, lam=lam)

df["covered_vanilla"] = cov_v
df["setsize_vanilla"] = size_v
df["singlecls_vanilla"] = single_v  # 0 or 1 if single, NaN otherwise

df["covered_linear"] = cov_linear
df["setsize_linear"] = size_linear
df["singlecls_linear"] = single_linear

# Correctness for singles (only defined if set size == 1)
df["correct_vanilla_single"] = ((df["setsize_vanilla"] == 1) & (df["singlecls_vanilla"] == df["gt"].astype(float))).astype(int)
df["correct_linear_single"]    = ((df["setsize_linear"] == 1) & (df["singlecls_linear"] == df["gt"].astype(float))).astype(int)

# Expansion flag
df["expanded_linear"] = (df["covered_linear"] == 1) & (df["covered_vanilla"] == 0)

# ---- Choose focus points for cropping ----
if isinstance(FOCUS, tuple) and FOCUS[0] == "block":
    blk = int(FOCUS[1])
    pts = df[df["block_id"] == blk].copy() if "block_id" in df.columns else df.copy()
elif FOCUS == "expanded":
    pts = df[df["expanded_linear"]].copy()
    if len(pts) == 0:  # fallback
        pts = df.copy()
elif FOCUS == "high_var":
    qv = df["Vn"].quantile(0.90)
    pts = df[df["Vn"] >= qv].copy()
else:
    pts = df.copy()

xs = pts["x"].values; ys = pts["y"].values
xmin, xmax = xs.min() - MARGIN, xs.max() + MARGIN
ymin, ymax = ys.min() - MARGIN, ys.max() + MARGIN

# ---- Windowed read + downsample basemap ----
with rasterio.open(RASTER_PATH) as ds:
    xmin = max(xmin, ds.bounds.left);  xmax = min(xmax, ds.bounds.right)
    ymin = max(ymin, ds.bounds.bottom); ymax = min(ymax, ds.bounds.top)

    win = from_bounds(xmin, ymin, xmax, ymax, ds.transform).round_offsets().round_lengths()
    h = int(win.height); w = int(win.width)
    scale = max(h, w) / float(TARGET_LONG_SIDE) if max(h, w) > TARGET_LONG_SIDE else 1.0
    out_h = max(1, int(h / scale)); out_w = max(1, int(w / scale))

    rgb = ds.read([1,2,3], window=win, out_shape=(3, out_h, out_w), resampling=Resampling.bilinear).astype(np.float32)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-12)

    # Window transform at downsampled resolution
    win_transform = ds.window_transform(win)
    win_transform = win_transform * win_transform.scale(win.width / float(out_w), win.height / float(out_h))
    x0, y0 = win_transform * (0, 0)
    x1, y1 = win_transform * (out_w, out_h)
    extent = (x0, x1, y1, y0)  # origin='upper'

# ---- Subset points to crop window ----
inside = (df["x"].between(xmin, xmax)) & (df["y"].between(ymin, ymax))
D = df.loc[inside].copy()

# ---- Plot helpers ----
# ========================= Improved Figures =========================
# Tuning knobs (easy to tweak for readability)
SIZE_BASE   = 10   # most points
SIZE_SINGLE = 16   # singletons
SIZE_SPECIAL= 28   # empty & two-label & expansion rings
ALPHA_BASE  = 0.55
ALPHA_SINGLE= 0.75

# --- Helper: nice legend outside the axes
def place_shared_legend(fig, ax_for_handles, loc="lower center"):
    handles, labels = ax_for_handles.get_legend_handles_labels()
    fig.legend(handles, labels, loc=loc, ncol=4, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(bottom=0.12)  # make room

# --- Plot helpers (draw rare categories on top, with larger markers)
def plot_panel(ax, title, base_img, extent, points, mode):
    ax.imshow(np.transpose(base_img, (1,2,0)), extent=extent, origin="upper")
    ax.set_title(title, pad=8)
    ax.set_xlabel("Easting"); ax.set_ylabel("Northing")
    ax.set_aspect('equal', adjustable='box')

    if mode == "vanilla":
        sizecol, singlecol, correctcol = "setsize_vanilla", "singlecls_vanilla", "correct_vanilla_single"
    elif mode == "linear":
        sizecol, singlecol, correctcol = "setsize_linear", "singlecls_linear", "correct_linear_single"
    else:
        raise ValueError

    # 1) Majority: draw singles first (underneath) so specials pop
    m = (points[sizecol] == 1) & (points[correctcol] == 1)             # single + correct
    ax.scatter(points.loc[m,"x"], points.loc[m,"y"],
               marker='o', s=SIZE_SINGLE, c="#20a320", alpha=ALPHA_SINGLE,
               linewidths=0, label="Singleton (correct)", zorder=2)

    m = (points[sizecol] == 1) & (points[correctcol] == 0)             # single + incorrect
    ax.scatter(points.loc[m,"x"], points.loc[m,"y"],
               marker='o', s=SIZE_SINGLE, c="#cc3737", alpha=ALPHA_SINGLE,
               linewidths=0, label="Singleton (incorrect)", zorder=2)

    # 2) Rare types: draw on top, bigger, with clear outlines
    m = points[sizecol] == 0                                           # empty
    ax.scatter(points.loc[m,"x"], points.loc[m,"y"],
               marker='x', s=SIZE_SPECIAL, c="black", alpha=0.9,
               linewidths=1.4, label="Empty set", zorder=3)

    m = points[sizecol] == 2                                           # two-label
    ax.scatter(points.loc[m,"x"], points.loc[m,"y"],
               marker='s', s=SIZE_SPECIAL, facecolors='none', edgecolors="black",
               linewidths=1.6, label="Two-label set", zorder=3)

    # 3) For linear panel, add expansion rings
    if mode == "linear":
        exp = points["expanded_linear"] == 1
        ax.scatter(points.loc[exp,"x"], points.loc[exp,"y"],
                   s=SIZE_SPECIAL*1.4, facecolors='none', edgecolors='white', linewidths=1.8,
                   label="Expansion (lambda=3)", zorder=4)
        ax.scatter(points.loc[exp,"x"], points.loc[exp,"y"],
                   s=SIZE_SPECIAL*1.4, facecolors='none', edgecolors='black', linewidths=0.7,
                   zorder=4)

# --- Build improved triptych
fig, axes = plt.subplots(1, 3, figsize=(16, 6), constrained_layout=False)

# A) Ground truth
axes[0].imshow(np.transpose(rgb, (1,2,0)), extent=extent, origin="upper")
axes[0].set_title("Ground Truth (points)", pad=8)
axes[0].set_xlabel("Easting"); axes[0].set_ylabel("Northing"); axes[0].set_aspect('equal', adjustable='box')
m0 = D["gt"] == 0; m1 = D["gt"] == 1
axes[0].scatter(D.loc[m0,"x"], D.loc[m0,"y"], s=SIZE_BASE, c="#7b68ee", alpha=0.7, label="GT=0", zorder=2)
axes[0].scatter(D.loc[m1,"x"], D.loc[m1,"y"], s=SIZE_BASE, c="#2ca02c", alpha=0.7, label="GT=1", zorder=2)
axes[0].legend(loc="lower right", fontsize=8)

# B) Vanilla
plot_panel(axes[1], "Vanilla CP - set type & correctness", rgb, extent, D, mode="vanilla")

# C) Linear lambda=3
plot_panel(axes[2], "Linear CP (lambda=3) - set type & correctness", rgb, extent, D, mode="linear")

# Move the shared legend outside to avoid overlap
place_shared_legend(fig, axes[2])

fig.suptitle("Cropped Spatial Comparison: GT vs Vanilla vs Linear (lambda=3)", y=0.99, fontsize=15)
triptych_path = os.path.join(OUT_DIR, "overlay_triptych_GT_v_Vanilla_v_Linear_l3_CROPPED_improved.png")
plt.savefig(triptych_path, dpi=220, bbox_inches="tight")
plt.close()
print(f"   {triptych_path}")

# ========================= Delta-view: show only CHANGES =========================
# Category encoding for changes Vanilla -> Linear (lambda=3)
# we only plot points whose coverage or set-size changed; color encodes the outcome
delta = D.copy()
delta["changed"] = (delta["setsize_vanilla"] != delta["setsize_linear"]) | (delta["covered_vanilla"] != delta["covered_linear"])

chg = delta[delta["changed"]].copy()

# classify change:
#   gained_covered_correct   : vanilla empty -> linear single(correct)
#   gained_covered_incorrect : vanilla empty -> linear single(incorrect)
#   gained_two               : any -> linear two-label (optional if showing multi-sets)
#   lost_coverage            : vanilla covered -> linear empty (rare with linear)
chg["label"] = "other"
# gained coverage (vanilla empty -> linear covered)
gc = (delta["covered_vanilla"] == 0) & (delta["covered_linear"] == 1)
g_single = gc & (delta["setsize_linear"] == 1)
chg.loc[g_single & (delta["correct_linear_single"] == 1), "label"] = "gained_single_correct"
chg.loc[g_single & (delta["correct_linear_single"] == 0), "label"] = "gained_single_incorrect"
chg.loc[gc & (delta["setsize_linear"] == 2), "label"] = "gained_two"

# lost coverage (should be very rare)
lc = (delta["covered_vanilla"] == 1) & (delta["covered_linear"] == 0)
chg.loc[lc, "label"] = "lost_coverage"

# map to colors/markers
STYLE = {
    "gained_single_correct":   dict(color="#1b9e77", marker="o",  s=28, alpha=0.9, lw=0),
    "gained_single_incorrect": dict(color="#d95f02", marker="o",  s=28, alpha=0.9, lw=0),
    "gained_two":              dict(edgecolor="#7570b3", facecolors="none", marker="s", s=34, alpha=0.95, lw=1.6),
    "lost_coverage":           dict(color="#e7298a", marker="x",  s=34, alpha=0.95, lw=1.6),
}

fig, ax = plt.subplots(figsize=(7.8, 6))
ax.imshow(np.transpose(rgb, (1,2,0)), extent=extent, origin="upper")
ax.set_title("Coverage Changes: Vanilla -> Linear (lambda=3)\n(only points that changed)", pad=10)
ax.set_xlabel("Easting"); ax.set_ylabel("Northing"); ax.set_aspect('equal', adjustable='box')

for key, style in STYLE.items():
    m = chg["label"] == key
    if m.any():
        if "edgecolor" in style:  # two-label gained
            ax.scatter(chg.loc[m,"x"], chg.loc[m,"y"], zorder=3, **style, label=f"{key.replace('_',' ')} (n={m.sum()})")
        else:
            ax.scatter(chg.loc[m,"x"], chg.loc[m,"y"], zorder=3, **style, label=f"{key.replace('_',' ')} (n={m.sum()})")

ax.legend(loc="upper right", frameon=True)
delta_path = os.path.join(OUT_DIR, "overlay_delta_changes_Vanilla_to_Linear_l3_CROPPED.png")
plt.tight_layout()
plt.savefig(delta_path, dpi=220, bbox_inches="tight")
plt.close()
print(f"   {delta_path}")

