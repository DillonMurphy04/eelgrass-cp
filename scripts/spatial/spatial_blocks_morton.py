"""
Spatial-blocked evaluation on 2022 points using Morton (Z-order) grouping.
Compares vanilla, linear lambda=3, and nonparametric; outputs block summaries and figures in OUT_DIR.
"""

import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from pyproj import CRS, Transformer
from src.eelgrass_cp import qhat_conformal


# ------------------------- User Config -------------------------
CACHE_2022_NPZ = r"D:\2022 Data\cp_spatial_blocks_2022\points_2022_cache_with_xy.npz"
CACHE_2021_NPZ = r"D:\4 year training data\2021\ensemble_2020_2021_pixel_calib_cache\pixel_calib_cache_2021.npz"
META_JSON = r"D:\4 year training data\2021\ensemble_2020_2021_pixel_calib_cache\pixel_calib_meta_2021.json"

OUT_DIR = r"D:\2022 Data\cp_spatial_blocks_2022_linear_nonparam_MORTON"
os.makedirs(OUT_DIR, exist_ok=True)

MIRROR_DIR = r"D:\2022 Data\cp_spatial_blocks_2022"
MIRROR_OUTPUTS = True

N_BLOCKS = 10
SEED = 42

LAMBDA_LINEAR = 3.0

FIG_DPI = 220

# Morton settings (coarse-cell grouping to improve contiguity)
# p_coarse controls grid resolution: grid = 2^p x 2^p
# Smaller p_coarse => coarser cells => more contiguous blocks but less precise boundaries.
P_COARSE = 11  # 2^11 = 2048 cells per axis


# ------------------------- Helpers -------------------------
def normalize_var(v, vmin, vmax):
    return np.clip((v - vmin) / max(vmax - vmin, 1e-12), 0.0, 1.0)

def score_transform_linear(s, vnorm, lam):
    return s / (1.0 + lam * vnorm)

def set_composition_binary(p0, p1, q, mode="vanilla", vnorm=None, lam=None, sigma_fn=None):
    s0 = 1.0 - p0
    s1 = 1.0 - p1
    if mode == "linear":
        s0 = score_transform_linear(s0, vnorm, lam)
        s1 = score_transform_linear(s1, vnorm, lam)
    elif mode == "nonparametric":
        s0 = s0 / sigma_fn(vnorm)
        s1 = s1 / sigma_fn(vnorm)

    k = (s0 <= q).astype(np.int32) + (s1 <= q).astype(np.int32)
    n = len(k)
    e = int(np.sum(k == 0))
    s = int(np.sum(k == 1))
    t = n - e - s
    return {
        "n": n, "empty": e, "single": s, "two": t,
        "pct_empty": e / n if n else np.nan,
        "pct_single": s / n if n else np.nan,
        "pct_two": t / n if n else np.nan
    }

def softmax_rows(logits, T):
    z = logits / max(T, 1e-8)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.clip(e.sum(axis=1, keepdims=True), 1e-12, None)

def compute_sigma_and_qhat_nonparametric(meta_json_path, cache_2021_npz, bins=20):
    from sklearn.model_selection import train_test_split
    from scipy.interpolate import interp1d

    with open(meta_json_path, "r") as f:
        meta = json.load(f)

    T = float(meta["temperature_T"])
    V_MIN = float(meta["variance_min"])
    V_MAX = float(meta["variance_max"])
    ALPHA = float(meta["alpha"])
    SEED_LOCAL = int(meta.get("seed", 42))
    CAL_TEST_SPLIT = float(meta["splits"]["CAL_TEST_SPLIT"])
    CAL_TUNE_FRAC = float(meta["splits"]["CAL_TUNE_FRAC"])

    D = np.load(cache_2021_npz, allow_pickle=True)
    PLOG_all = D["PLOG"]
    Y_all = D["Y"]
    V_all = D["V"]
    idx_cp = D.get("idx_cp", None)
    pool = np.arange(len(Y_all)) if idx_cp is None else idx_cp

    P_cp = softmax_rows(PLOG_all[pool], T=T).astype(np.float32)
    Y_cp = Y_all[pool].astype(np.int64)
    V_cp = V_all[pool].astype(np.float64)

    Np = len(P_cp)
    py = P_cp[np.arange(Np), Y_cp]
    s = 1.0 - py
    Vn = normalize_var(V_cp, V_MIN, V_MAX)

    idx_all = np.arange(Np)
    cal_idx, _ = train_test_split(idx_all, test_size=CAL_TEST_SPLIT, random_state=SEED_LOCAL, shuffle=True)
    cal_sub_idx, _ = train_test_split(cal_idx, test_size=CAL_TUNE_FRAC, random_state=SEED_LOCAL, shuffle=True)

    df_cal = pd.DataFrame({"var": Vn[cal_idx], "score": s[cal_idx]})
    df_cal["var_bin"] = pd.qcut(df_cal["var"], q=bins, duplicates="drop")

    bin_means = df_cal.groupby("var_bin", observed=False)["score"].mean().reset_index()
    centers = np.array([iv.mid for iv in bin_means["var_bin"]], dtype=float)
    sigma = bin_means["score"].values.astype(float)

    sigma_fn = interp1d(centers, sigma, kind="linear", fill_value="extrapolate", assume_sorted=False)

    cal_scores_norm = s[cal_idx] / sigma_fn(Vn[cal_idx])
    pos_in_cal = {g: i for i, g in enumerate(cal_idx)}
    mask = np.zeros(len(cal_idx), dtype=bool)
    for g in cal_sub_idx:
        mask[pos_in_cal[g]] = True

    qhat = qhat_conformal(cal_scores_norm[mask], ALPHA)
    return sigma_fn, qhat

def choose_utm_transformer(lons, lats):
    lon = float(np.median(lons))
    lat = float(np.median(lats))
    zone = int(np.floor((lon + 180.0) / 6.0) + 1)
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    crs_utm = CRS.from_epsg(epsg)
    transformer = Transformer.from_crs("EPSG:4326", crs_utm, always_xy=True)
    return crs_utm, transformer

def to_utm_meters(lons, lats):
    crs_utm, tr = choose_utm_transformer(lons, lats)
    E, N = tr.transform(lons, lats)
    return crs_utm, np.asarray(E, dtype=np.float64), np.asarray(N, dtype=np.float64)

def km_offset(E_m, N_m):
    E0 = float(np.min(E_m))
    N0 = float(np.min(N_m))
    Ex = (E_m - E0) / 1000.0
    Ny = (N_m - N0) / 1000.0
    return Ex, Ny, E0, N0

def _quantize_to_grid(E_m, N_m, p):
    n = 1 << p
    Emin, Emax = float(np.min(E_m)), float(np.max(E_m))
    Nmin, Nmax = float(np.min(N_m)), float(np.max(N_m))
    dx = max(Emax - Emin, 1e-12)
    dy = max(Nmax - Nmin, 1e-12)
    xi = np.floor((E_m - Emin) / dx * (n - 1)).astype(np.int64)
    yi = np.floor((N_m - Nmin) / dy * (n - 1)).astype(np.int64)
    xi = np.clip(xi, 0, n - 1)
    yi = np.clip(yi, 0, n - 1)
    return xi, yi

def _part1by1(n):
    n &= 0xFFFFFFFF
    n = (n | (n << 16)) & 0x0000FFFF0000FFFF
    n = (n | (n << 8))  & 0x00FF00FF00FF00FF
    n = (n | (n << 4))  & 0x0F0F0F0F0F0F0F0F
    n = (n | (n << 2))  & 0x3333333333333333
    n = (n | (n << 1))  & 0x5555555555555555
    return n

def morton_code_2d(xi, yi):
    xi = xi.astype(np.uint64)
    yi = yi.astype(np.uint64)
    x = np.vectorize(_part1by1, otypes=[np.uint64])(xi)
    y = np.vectorize(_part1by1, otypes=[np.uint64])(yi)
    return (x | (y << 1)).astype(np.uint64)

def morton_coarse_cell_order(E_m, N_m, p_coarse=11):
    xi, yi = _quantize_to_grid(E_m, N_m, p=p_coarse)
    cell_code = morton_code_2d(xi, yi)
    order_cells = np.argsort(cell_code, kind="mergesort")
    return order_cells, cell_code

def make_equal_count_blocks_from_cell_order(order, cell_code, n_blocks):
    n = len(order)
    base = n // n_blocks
    rem = n % n_blocks
    targets = np.array([base + (1 if b < rem else 0) for b in range(n_blocks)], dtype=int)

    uniq_codes, first_idx = np.unique(cell_code[order], return_index=True)
    cell_starts = np.sort(first_idx)
    cell_starts = np.append(cell_starts, n)

    block_id = -np.ones(n, dtype=np.int32)

    boundaries = [0]
    cum = 0
    t = targets[0]
    b = 0

    i = 0
    while i < len(cell_starts) - 1 and b < n_blocks - 1:
        cell_begin = cell_starts[i]
        cell_end = cell_starts[i + 1]
        cell_size = cell_end - cell_begin

        if cum + cell_size > t and cum > 0:
            boundaries.append(cell_begin)
            b += 1
            cum = 0
            t = targets[b]
            continue

        cum += cell_size
        i += 1

    boundaries.append(n)
    boundaries = np.array(boundaries, dtype=int)

    for b in range(n_blocks):
        start = boundaries[b]
        end = boundaries[b + 1]
        idx = order[start:end]
        block_id[idx] = b

    if np.any(block_id < 0):
        raise RuntimeError("Block assignment failed (unassigned points).")

    sizes = np.bincount(block_id, minlength=n_blocks)
    return block_id, sizes, targets, boundaries

def discrete_colors(n):
    cmap = plt.get_cmap("tab10", n)
    return [cmap(i) for i in range(n)]

def savefig(path):
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()

def as_pct(x):
    return f"{x*100:.1f}%" if isinstance(x, (int, float, np.floating)) and np.isfinite(x) else "nan"


# ------------------------- Load meta -------------------------
print(" Loading calibration meta...")
with open(META_JSON, "r") as f:
    meta = json.load(f)

ALPHA = float(meta["alpha"])
TARGET = 1.0 - ALPHA
T = float(meta["temperature_T"])
V_MIN = float(meta["variance_min"])
V_MAX = float(meta["variance_max"])

qhat_vanilla = float(meta["pixel_level"]["vanilla"]["qhat"])

qhat_linear = None
adaptive = meta.get("pixel_level", {}).get("adaptive", {})
linear_pack = adaptive.get("linear", adaptive.get("normalized", {}))
for lam_str, pack in linear_pack.items():
    if abs(float(lam_str) - float(LAMBDA_LINEAR)) < 1e-9:
        qhat_linear = float(pack["qhat"])
        break
if qhat_linear is None:
    raise RuntimeError(f"q for linear lambda={LAMBDA_LINEAR} not found in meta JSON.")

print(f"  alpha={ALPHA:.3f}  target={TARGET:.3f}  T={T:.3f}  Vmin={V_MIN:.3e}  Vmax={V_MAX:.3e}")
print(f"  q: vanilla={qhat_vanilla:.4f}, linear(lambda={LAMBDA_LINEAR:g})={qhat_linear:.4f}")

print(" Building nonparametric sigma(Vn) and q from 2021 calibration...")
sigma_fn, qhat_nonparametric = compute_sigma_and_qhat_nonparametric(META_JSON, CACHE_2021_NPZ, bins=20)
print(f"  q_nonparametric (from 2021 cal_sub) = {qhat_nonparametric:.4f}")


# ------------------------- Load 2022 cache -------------------------
print(f" Loading 2022 cache: {CACHE_2022_NPZ}")
D22 = np.load(CACHE_2022_NPZ, allow_pickle=True)
required = ["p0", "p1", "V_center", "y", "ids", "xs", "ys"]
if not all(k in D22 for k in required):
    missing = [k for k in required if k not in D22]
    raise RuntimeError(f"2022 cache missing fields: {missing}")

p0 = D22["p0"].astype(np.float32)
p1 = D22["p1"].astype(np.float32)
Vc = D22["V_center"].astype(np.float64)
y = D22["y"].astype(np.int64)
ids = D22["ids"].astype(np.int64)
xs = D22["xs"].astype(np.float64)
ys = D22["ys"].astype(np.float64)

N = len(ids)
print(f"  N points: {N}")
if N < N_BLOCKS:
    raise RuntimeError("Not enough points to create blocks.")


# ------------------------- Project to UTM (meters) -------------------------
crs_utm, E_m, N_m = to_utm_meters(xs, ys)
print(f" Using UTM CRS: {crs_utm.to_string()}")


# ------------------------- Morton coarse-cell equal-count blocks -------------------------
print(f" Morton coarse-cell ordering (p={P_COARSE}, grid={1<<P_COARSE}x{1<<P_COARSE})...")
order, cell_code = morton_coarse_cell_order(E_m, N_m, p_coarse=P_COARSE)
block_ids, sizes, targets, boundaries = make_equal_count_blocks_from_cell_order(order, cell_code, N_BLOCKS)

print(" Block sizes:")
for b in range(N_BLOCKS):
    print(f"  Block {b}: size={int(sizes[b])} (target={int(targets[b])})")
print(f" Boundary indices in Morton-sorted list: {boundaries.tolist()}")
if np.max(np.abs(sizes - targets)) > 0:
    print(" Note: exact equal counts may not be possible when preserving whole Morton cells.")


# ------------------------- Scores and coverage -------------------------
Vn = normalize_var(Vc, V_MIN, V_MAX)
py = np.where(y == 1, p1, p0).astype(np.float64)
base_scores = 1.0 - py

covered_vanilla = (base_scores <= qhat_vanilla).astype(np.int32)
scores_linear = score_transform_linear(base_scores, Vn, LAMBDA_LINEAR)
covered_linear = (scores_linear <= qhat_linear).astype(np.int32)
scores_nonparam = base_scores / sigma_fn(Vn)
covered_nonparam = (scores_nonparam <= qhat_nonparametric).astype(np.int32)

sc_van = set_composition_binary(p0, p1, qhat_vanilla, mode="vanilla")
sc_lin = set_composition_binary(p0, p1, qhat_linear, mode="linear", vnorm=Vn, lam=LAMBDA_LINEAR)
sc_nonparam = set_composition_binary(p0, p1, qhat_nonparametric, mode="nonparametric", vnorm=Vn, sigma_fn=sigma_fn)


# ------------------------- DataFrame + per-block metrics -------------------------
df = pd.DataFrame({
    "id": ids,
    "gt": y,
    "lon": xs,
    "lat": ys,
    "E_m": E_m,
    "N_m": N_m,
    "p0": p0,
    "p1": p1,
    "V": Vc,
    "Vn": Vn,
    "base_score": base_scores,
    "covered_vanilla": covered_vanilla,
    "covered_linear": covered_linear,
    "covered_nonparam": covered_nonparam,
    "block_id": block_ids,
    "morton_cell": cell_code.astype(np.uint64)
})

def block_metrics(group):
    idx = group.index.values
    sc_v = set_composition_binary(p0[idx], p1[idx], qhat_vanilla, mode="vanilla")
    sc_l = set_composition_binary(p0[idx], p1[idx], qhat_linear, mode="linear", vnorm=Vn[idx], lam=LAMBDA_LINEAR)
    sc_h = set_composition_binary(p0[idx], p1[idx], qhat_nonparametric, mode="nonparametric", vnorm=Vn[idx], sigma_fn=sigma_fn)
    return pd.Series({
        "n_points": len(group),
        "coverage_vanilla": float(group["covered_vanilla"].mean()),
        "coverage_linear": float(group["covered_linear"].mean()),
        "coverage_nonparam": float(group["covered_nonparam"].mean()),
        "singleton_vanilla": sc_v["pct_single"],
        "empty_vanilla": sc_v["pct_empty"],
        "two_vanilla": sc_v["pct_two"],
        "singleton_linear": sc_l["pct_single"],
        "empty_linear": sc_l["pct_empty"],
        "two_linear": sc_l["pct_two"],
        "singleton_nonparam": sc_h["pct_single"],
        "empty_nonparam": sc_h["pct_empty"],
        "two_nonparam": sc_h["pct_two"],
        "mean_var": float(group["V"].mean()),
        "mean_vn": float(group["Vn"].mean()),
        "E_mean": float(group["E_m"].mean()),
        "N_mean": float(group["N_m"].mean())
    })

block_summary = df.groupby("block_id", sort=True).apply(block_metrics).reset_index()
block_summary_path = os.path.join(OUT_DIR, "block_summary.csv")
block_summary.to_csv(block_summary_path, index=False)

cov_sd_van = float(block_summary["coverage_vanilla"].std(ddof=1))
cov_sd_lin = float(block_summary["coverage_linear"].std(ddof=1))
cov_sd_non = float(block_summary["coverage_nonparam"].std(ddof=1))

r_van, p_van = pearsonr(block_summary["mean_var"], block_summary["coverage_vanilla"])
r_lin, p_lin = pearsonr(block_summary["mean_var"], block_summary["coverage_linear"])
r_non, p_non = pearsonr(block_summary["mean_var"], block_summary["coverage_nonparam"])

summary = {
    "alpha": ALPHA,
    "target": TARGET,
    "temperature_T": T,
    "variance_norm": {"min": V_MIN, "max": V_MAX},
    "counts": {
        "N_points": int(N),
        "n_class0": int(np.sum(y == 0)),
        "n_class1": int(np.sum(y == 1)),
        "blocks": int(N_BLOCKS)
    },
    "blocking": {
        "method": "morton_coarse_cell_equal_count",
        "p_coarse": int(P_COARSE),
        "grid_size": int(1 << P_COARSE),
        "targets": targets.tolist(),
        "sizes": sizes.tolist(),
        "utm_crs": crs_utm.to_string()
    },
    "methods": {
        "vanilla": {
            "qhat": float(qhat_vanilla),
            "global": {
                "coverage": float(np.mean(base_scores <= qhat_vanilla)),
                "set_composition": sc_van
            },
            "coverage_across_blocks": {"sd": cov_sd_van}
        },
        "linear": {
            "lambda": float(LAMBDA_LINEAR),
            "qhat": float(qhat_linear),
            "global": {
                "coverage": float(np.mean(scores_linear <= qhat_linear)),
                "set_composition": sc_lin
            },
            "coverage_across_blocks": {"sd": cov_sd_lin}
        },
        "nonparametric": {
            "qhat": float(qhat_nonparametric),
            "global": {
                "coverage": float(np.mean(scores_nonparam <= qhat_nonparametric)),
                "set_composition": sc_nonparam
            },
            "coverage_across_blocks": {"sd": cov_sd_non}
        }
    },
    "spatial_correlation": {
        "variance_vs_coverage_vanilla": {"r": float(r_van), "p": float(p_van)},
        "variance_vs_coverage_linear": {"r": float(r_lin), "p": float(p_lin)},
        "variance_vs_coverage_nonparam": {"r": float(r_non), "p": float(p_non)}
    },
    "files": {
        "per_point_csv": "points_with_blocks.csv",
        "block_summary_csv": Path(block_summary_path).name
    }
}

summary_path = os.path.join(OUT_DIR, "summary.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

per_point_path = os.path.join(OUT_DIR, "points_with_blocks.csv")
df.to_csv(per_point_path, index=False)


# ------------------------- Figures -------------------------
colors = discrete_colors(N_BLOCKS)
Ex_km, Ny_km, E0, N0 = km_offset(E_m, N_m)

plt.figure(figsize=(8.8, 9.2), dpi=FIG_DPI)
ax = plt.gca()
for b in range(N_BLOCKS):
    m = (block_ids == b)
    ax.scatter(
        Ex_km[m], Ny_km[m],
        s=32,
        c=[colors[b]],
        alpha=0.90,
        edgecolors="black",
        linewidths=0.25,
        label=f"Block {b}"
    )
ax.set_xlabel(f"Easting (km, offset +{int(round(E0))} m)")
ax.set_ylabel(f"Northing (km, offset +{int(round(N0))} m)")
ax.set_title(f"Spatial blocks (Morton coarse-cell order in UTM; {N_BLOCKS} equal-count groups)", pad=12)
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, title="block_id", markerscale=1.1)
savefig(os.path.join(OUT_DIR, "blocks_scatter.png"))

plt.figure(figsize=(9.0, 4.4), dpi=FIG_DPI)
plt.bar(block_summary["block_id"].astype(str), block_summary["mean_var"].values)
plt.xlabel("Block ID")
plt.ylabel("Mean variance (raw)")
plt.title("Per-block Mean Variance (2022 points)")
savefig(os.path.join(OUT_DIR, "block_variance_bar.png"))

x_ids = block_summary["block_id"].astype(str).values
x_pos = np.arange(len(x_ids))
w = 0.26
plt.figure(figsize=(11.0, 4.6), dpi=FIG_DPI)
plt.bar(x_pos - w, block_summary["coverage_vanilla"].values, width=w, label="Vanilla")
plt.bar(x_pos,      block_summary["coverage_linear"].values,  width=w, label=f"Linear (lambda={LAMBDA_LINEAR:g})")
plt.bar(x_pos + w,  block_summary["coverage_nonparam"].values, width=w, label="Nonparametric")
plt.axhline(TARGET, linestyle="--", linewidth=1)
plt.xticks(x_pos, x_ids)
plt.ylim(0, 1)
plt.xlabel("Block ID")
plt.ylabel("Coverage")
plt.title("Per-block Coverage")
plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
savefig(os.path.join(OUT_DIR, "block_coverage_compare.png"))

# ========================= Block-wise Coverage vs Mean Variance =========================
# Zoomed y-axis + best-fit lines (no equation labels)

xv = block_summary["mean_var"].values.astype(float)

yv_v = block_summary["coverage_vanilla"].values.astype(float)
yv_l = block_summary["coverage_linear"].values.astype(float)
yv_n = block_summary["coverage_nonparam"].values.astype(float)

# Least-squares fits
a_v, b_v = np.polyfit(xv, yv_v, 1)
a_l, b_l = np.polyfit(xv, yv_l, 1)
a_n, b_n = np.polyfit(xv, yv_n, 1)

xgrid = np.linspace(xv.min(), xv.max(), 300)

# Zoom y-limits to data range with padding
all_y = np.concatenate([yv_v, yv_l, yv_n])
pad = 0.03
ymin = max(0.0, float(all_y.min() - pad))
ymax = min(1.0, float(all_y.max() + pad))

plt.figure(figsize=(7.8, 6.0), dpi=FIG_DPI)

# Scatter points
plt.scatter(
    xv, yv_v, s=85, alpha=0.9,
    edgecolors="black", linewidths=0.25,
    label="Vanilla"
)
plt.scatter(
    xv, yv_l, s=85, alpha=0.9,
    edgecolors="black", linewidths=0.25,
    label="Linear (lambda=3)"
)
plt.scatter(
    xv, yv_n, s=85, alpha=0.9,
    edgecolors="black", linewidths=0.25,
    label="Nonparametric"
)

# Best-fit lines
plt.plot(xgrid, a_v * xgrid + b_v, linewidth=2.2)
plt.plot(xgrid, a_l * xgrid + b_l, linewidth=2.2)
plt.plot(xgrid, a_n * xgrid + b_n, linewidth=2.2)

plt.xlabel("Mean Variance per Block")
plt.ylabel("Coverage")
plt.ylim(ymin, ymax)
plt.title("Block-wise Coverage vs Mean Variance (2022)")

plt.legend(
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    frameon=True
)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "block_cov_vs_var_scatter.png"), bbox_inches="tight")
plt.close()


export_cols = [
    "block_id", "n_points",
    "mean_var", "mean_vn",
    "coverage_vanilla", "coverage_linear", "coverage_nonparam",
    "singleton_vanilla", "singleton_linear", "singleton_nonparam",
    "empty_vanilla", "empty_linear", "empty_nonparam"
]
appendix_csv = os.path.join(OUT_DIR, "appendix_block_metrics.csv")
block_summary.to_csv(appendix_csv, columns=export_cols, index=False)

latex_table = block_summary[export_cols].to_latex(index=False, float_format="%.3f")
latex_path = os.path.join(OUT_DIR, "appendix_block_metrics.tex")
with open(latex_path, "w") as f:
    f.write(latex_table)


# ------------------------- Console report -------------------------
print("\n Spatial-blocked results on 2022 points")
print(f"Global vanilla:   cov={summary['methods']['vanilla']['global']['coverage']:.3f} | set: single={as_pct(sc_van['pct_single'])} empty={as_pct(sc_van['pct_empty'])} two={as_pct(sc_van['pct_two'])}")
print(f"Global linear:    cov={summary['methods']['linear']['global']['coverage']:.3f} | set: single={as_pct(sc_lin['pct_single'])} empty={as_pct(sc_lin['pct_empty'])} two={as_pct(sc_lin['pct_two'])}")
print(f"Global nonparam:  cov={summary['methods']['nonparametric']['global']['coverage']:.3f} | set: single={as_pct(sc_nonparam['pct_single'])} empty={as_pct(sc_nonparam['pct_empty'])} two={as_pct(sc_nonparam['pct_two'])}")

print("\nPer-block coverage SD:")
print(f"  Vanilla:          SD={cov_sd_van:.4f}")
print(f"  Linear (lambda={LAMBDA_LINEAR:g}):     SD={cov_sd_lin:.4f}")
print(f"  Nonparametric:    SD={cov_sd_non:.4f}")

print("\n Correlation diagnostics (mean variance vs coverage):")
print(f"  Vanilla:          r={r_van:+.3f}, p={p_van:.4f}")
print(f"  Linear:           r={r_lin:+.3f}, p={p_lin:.4f}")
print(f"  Nonparametric:    r={r_non:+.3f}, p={p_non:.4f}")

print(f"\n Outputs saved to: {OUT_DIR}")
print(f"  per-point CSV   : {per_point_path}")
print(f"  block summary   : {block_summary_path}")
print(f"  summary JSON    : {summary_path}")
print(f"  appendix CSV    : {appendix_csv}")
print(f"  appendix TEX    : {latex_path}")
print(f"  figures         : blocks_scatter.png, block_variance_bar.png, block_coverage_compare.png, block_cov_vs_var_scatter.png")

if MIRROR_OUTPUTS:
    try:
        os.makedirs(MIRROR_DIR, exist_ok=True)
        mirror_files = [
            "points_with_blocks.csv",
            "block_summary.csv",
            "summary.json",
            "appendix_block_metrics.csv",
            "appendix_block_metrics.tex",
            "blocks_scatter.png",
            "block_variance_bar.png",
            "block_coverage_compare.png",
            "block_cov_vs_var_scatter.png",
        ]
        for fn in mirror_files:
            src = os.path.join(OUT_DIR, fn)
            dst = os.path.join(MIRROR_DIR, fn)
            if os.path.isfile(src):
                with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
                    fdst.write(fsrc.read())
        print(f"\n Mirrored key outputs into: {MIRROR_DIR}")
    except Exception as e:
        warnings.warn(f"Mirroring failed: {e}")
