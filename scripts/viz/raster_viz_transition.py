"""
Transition maps on the 2022 raster highlighting category changes from vanilla to linear/nonparametric.
Outputs a two-panel figure with a shared legend in OUT_DIR/viz_raster.
"""

import os, json
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import fiona
from shapely.geometry import shape
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
from src.eelgrass_cp import qhat_conformal, softmax_rows, normalize_var, fit_sigma_fn_from_cal, linear_score_transform

# -------- Config --------
RASTER_2022 = r"D:\\2022 Data\\Morro Bay Eelgrass AI - Rasterized Imagery\\2022_raster_final.tif"
POINTS_SHP  = r"D:\\2022 Data\\2022points\\2022points.shp"

OUT_DIR     = r"D:\\2022 Data\\cp_sensitivity_main"
FIGS_DIR    = os.path.join(OUT_DIR, "viz_raster")
os.makedirs(FIGS_DIR, exist_ok=True)

CACHE_2022_NPZ = os.path.join(OUT_DIR, "points_2022_cache_for_sensitivity.npz")
META_JSON      = r"D:\\4 year training data\\2021\\ensemble_2020_2021_pixel_calib_cache\\pixel_calib_meta_2021.json"
CACHE_2021_NPZ = r"D:\\4 year training data\\2021\\ensemble_2020_2021_pixel_calib_cache\\pixel_calib_cache_2021.npz"

ALPHA   = 0.10
FIG_DPI = 220
LAMBDA_LINEAR = 3.0

# -------- Helpers --------
def pretty_title(tag):
    if tag == "linear":
        return "Linear (lambda=3)"
    if tag == "nonparametric":
        return "Nonparametric"
    return tag

def compute_sigma_fn_and_qhats_from_2021(meta_json_path, cache_2021_npz,
                                         lam_linear=3.0, bins=20):
    """
    Reconstruct sigma(Vn) and q for:
      -  Nonparametric (nonparametric): s' = s / sigma(Vn)
      -  Linear (lambda=lam_linear):          s' = s / (1 + lambda Vn)
    using ONLY the 2021 CP cache + meta.
    """
    from sklearn.model_selection import train_test_split

    D = np.load(cache_2021_npz, allow_pickle=True)
    with open(meta_json_path, "r") as f:
        meta = json.load(f)

    alpha = float(meta["alpha"])
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

    # sigma(Vn) via quantile bins on cal_idx
    sigma_fn, _ = fit_sigma_fn_from_cal(Vn_cp, s_cp, cal_idx, bins=bins)

    # mask cal_sub within cal_idx
    pos_in_cal = {g: i for i, g in enumerate(cal_idx)}
    mask_cal = np.zeros(len(cal_idx), dtype=bool)
    for g in cal_sub_idx:
        if g in pos_in_cal:
            mask_cal[pos_in_cal[g]] = True

    s_cal  = s_cp[cal_idx]
    vn_cal = Vn_cp[cal_idx]

    # Nonparametric qhat
    s_nonparam_cal = s_cal / sigma_fn(vn_cal)
    qhat_nonparam = qhat_conformal(s_nonparam_cal[mask_cal], alpha)

    # Linear fallback qhat
    s_linear_cal = linear_score_transform(s_cal, vn_cal, lam_linear)
    qhat_linear_fallback = qhat_conformal(s_linear_cal[mask_cal], alpha)

    return sigma_fn, (V_MIN, V_MAX), qhat_nonparam, qhat_linear_fallback

# -------- Load per-point cache (2022) + meta (2021) --------
npz = np.load(CACHE_2022_NPZ)
p0 = npz["p0"].astype(np.float32)
p1 = npz["p1"].astype(np.float32)
Vc = npz["V_center"].astype(np.float32)
y  = npz["y"].astype(np.int64)

with open(META_JSON, "r") as f:
    meta = json.load(f)

V_MIN = float(meta["variance_min"])
V_MAX = float(meta["variance_max"])
Vn = normalize_var(Vc, V_MIN, V_MAX)

# Rebuild sigma_fn + qhats from 2021
sigma_fn, _, qhat_nonparam_2021, qhat_linear_fallback = compute_sigma_fn_and_qhats_from_2021(
    META_JSON, CACHE_2021_NPZ, lam_linear=LAMBDA_LINEAR, bins=20
)

# Vanilla qhat from meta
if "pixel_level" not in meta or "vanilla" not in meta["pixel_level"]:
    raise RuntimeError("Vanilla qhat not found in meta JSON.")
qh_van = float(meta["pixel_level"]["vanilla"]["qhat"])

# Linear lambda=3 qhat from meta if present else fallback
qh_linear = None
adaptive = meta.get("pixel_level", {}).get("adaptive", {})
linear_pack = adaptive.get("linear", adaptive.get("normalized", {}))
if linear_pack:
    for lam_str, pack in linear_pack.items():
        if abs(float(lam_str) - 3.0) < 1e-8:
            qh_linear = float(pack["qhat"])
            break
if qh_linear is None:
    qh_linear = qhat_linear_fallback

# Nonparametric qhat from 2021 reconstruction
qh_nonparam = qhat_nonparam_2021

# -------- Build category labels --------
# Label codes:
#   0 = empty
#   1 = singleton_correct
#   2 = singleton_wrong
#   3 = two-label
def build_label_map(p0, p1, y, qh, trans=None):
    p0s, p1s = 1 - p0, 1 - p1
    s0 = p0s if trans is None else trans(p0s)
    s1 = p1s if trans is None else trans(p1s)

    k = (s0 <= qh).astype(np.int8) + (s1 <= qh).astype(np.int8)

    pred_cls = np.where((s0 <= qh) & ~(s1 <= qh), 0,
               np.where((s1 <= qh) & ~(s0 <= qh), 1, -1)).astype(np.int8)

    lab = np.full(len(y), -1, dtype=np.int8)
    lab[k == 0] = 0
    lab[k == 2] = 3

    single = (k == 1)
    correct = single & (pred_cls == y)
    wrong   = single & (pred_cls != y)

    lab[correct] = 1
    lab[wrong]   = 2
    return lab

labels = {}
labels["vanilla"] = build_label_map(p0, p1, y, qh_van, trans=None)
labels["linear"] = build_label_map(p0, p1, y, qh_linear, trans=lambda s: linear_score_transform(s, Vn, LAMBDA_LINEAR))
labels["nonparametric"] = build_label_map(p0, p1, y, qh_nonparam, trans=lambda s: s / sigma_fn(Vn))

# -------- Read raster crop + point coords --------
with fiona.open(POINTS_SHP, "r") as src:
    xs, ys = [], []
    for feat in src:
        geom = shape(feat["geometry"])
        xs.append(geom.x); ys.append(geom.y)
xs = np.asarray(xs, dtype=np.float64)
ys = np.asarray(ys, dtype=np.float64)

xmin, xmax = xs.min(), xs.max()
ymin, ymax = ys.min(), ys.max()

def read_crop_rgb_with_extent(ds, xmin, ymin, xmax, ymax, target_long_side=2000):
    xmin = max(xmin, ds.bounds.left);  xmax = min(xmax, ds.bounds.right)
    ymin = max(ymin, ds.bounds.bottom); ymax = min(ymax, ds.bounds.top)
    win = from_bounds(xmin, ymin, xmax, ymax, ds.transform).round_offsets().round_lengths()

    h, w = int(win.height), int(win.width)
    scale = max(h, w) / float(target_long_side) if max(h, w) > target_long_side else 1.0
    out_h, out_w = max(1, int(h/scale)), max(1, int(w/scale))

    bands = [1,2,3] if ds.count >= 3 else [1,1,1]
    rgb = ds.read(
        bands, window=win, out_shape=(len(bands), out_h, out_w),
        resampling=Resampling.bilinear
    ).astype(np.float32)

    mn, mx = rgb.min(), rgb.max()
    if mx > mn:
        rgb = (rgb - mn) / (mx - mn + 1e-12)

    win_transform = ds.window_transform(win)
    win_transform = win_transform * win_transform.scale(win.width/float(out_w), win.height/float(out_h))
    x0, y0 = win_transform * (0, 0)
    x1, y1 = win_transform * (out_w, out_h)
    extent = (x0, x1, y1, y0)
    return rgb, extent, (xmin, ymin, xmax, ymax)

with rasterio.open(RASTER_2022) as ds:
    rgb, extent, (xmin, ymin, xmax, ymax) = read_crop_rgb_with_extent(ds, xmin, ymin, xmax, ymax)

inside = (xs >= xmin) & (xs <= xmax) & (ys >= ymin) & (ys <= ymax)

# -------- Plotting: updated buckets + shared legend --------
import matplotlib.gridspec as gridspec

def draw_base(ax, title):
    ax.imshow(np.transpose(rgb, (1,2,0)), extent=extent, origin="upper")
    ax.set_title(title, pad=2, fontsize=9)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

def compute_bucket_masks(to_tag):
    f = labels["vanilla"]
    t = labels[to_tag]
    changed = inside & (f != t)

    # 1) Empty or Incorrect -> Two-set
    m_twoset = changed & ((f == 0) | (f == 2)) & (t == 3)

    # 2) Empty -> Correct singleton
    m_emp2cor = changed & (f == 0) & (t == 1)

    # 3) Empty -> Incorrect singleton
    m_emp2inc = changed & (f == 0) & (t == 2)

    # 4) Other
    m_other = changed & ~(m_twoset | m_emp2cor | m_emp2inc)

    return m_twoset, m_emp2cor, m_emp2inc, m_other

def plot_buckets(ax, to_tag):
    m_twoset, m_emp2cor, m_emp2inc, m_other = compute_bucket_masks(to_tag)

    # 1) Empty/Incorrect -> Two-set  (TEAL SQUARE)
    ax.scatter(xs[m_twoset], ys[m_twoset],
               marker="s", s=70,
               facecolors="none", edgecolors="#17becf", lw=2.2,
               alpha=0.95, zorder=6)


    # 2) Empty -> Correct singleton (GREEN CIRCLE)
    ax.scatter(xs[m_emp2cor], ys[m_emp2cor],
               marker="o", s=48,
               c="#7ED957", lw=0,
               alpha=0.95, zorder=6)

    # 3) Empty -> Incorrect singleton (RED CIRCLE)
    ax.scatter(xs[m_emp2inc], ys[m_emp2inc],
               marker="o", s=48,
               c="#d62728", lw=0,
               alpha=0.95, zorder=6)

    # 4) Other (BLACK CIRCLE, same alpha)
    ax.scatter(xs[m_other], ys[m_other],
               marker="o", s=48,
               c="black", lw=0,
               alpha=0.95, zorder=2)

    return {
        "twoset": int(m_twoset.sum()),
        "emp2cor": int(m_emp2cor.sum()),
        "emp2inc": int(m_emp2inc.sum()),
        "other": int(m_other.sum()),
    }


# Layout: 2 panels + 1 legend row spanning both columns
fig = plt.figure(figsize=(9.2, 7.0))
gs = gridspec.GridSpec(2, 2, height_ratios=[12, 2.4], hspace=0.05, wspace=0.02)

axL = fig.add_subplot(gs[0, 0])
axR = fig.add_subplot(gs[0, 1])
axLeg = fig.add_subplot(gs[1, :])
axLeg.axis("off")

# Left: Linear
draw_base(axL, f"Transitions: Vanilla -> {pretty_title('linear')}")
counts_L = plot_buckets(axL, "linear")

# Right: Nonparametric
draw_base(axR, f"Transitions: Vanilla -> {pretty_title('nonparametric')}")
counts_R = plot_buckets(axR, "nonparametric")

# Shared legend with counts for BOTH panels
handles = [
    # teal square (hollow)
    plt.Line2D([0],[0], marker="s", linestyle="",
               markerfacecolor="none",
               markeredgecolor="#17becf",
               markeredgewidth=2.2, markersize=10),

    # green circle
    plt.Line2D([0],[0], marker="o", linestyle="",
               color="#7ED957", markersize=9),
    
    # red circle
    plt.Line2D([0],[0], marker="o", linestyle="",
               color="#d62728", markersize=9),

    # black circle
    plt.Line2D([0],[0], marker="o", linestyle="",
               color="black", markersize=9),
]

labels_legend = [
    f"Empty/Incorrect -> Two-set (Linear n={counts_L['twoset']}, Nonparametric n={counts_R['twoset']})",
    f"Empty -> Correct singleton (Linear n={counts_L['emp2cor']}, Nonparametric n={counts_R['emp2cor']})",
    f"Empty -> Incorrect singleton (Linear n={counts_L['emp2inc']}, Nonparametric n={counts_R['emp2inc']})",
    f"Other (Linear n={counts_L['other']}, Nonparametric n={counts_R['other']})",
]

axLeg.legend(handles, labels_legend, loc="center", ncol=1, frameon=True, fontsize=8)

fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.03)

outpath = os.path.join(FIGS_DIR, f"D_transitions_simplifiedPlus_a{ALPHA}_CROP.png")
plt.savefig(outpath, dpi=FIG_DPI, bbox_inches="tight")
plt.close()

print(" Exported:", outpath)
