"""
Visualize sigma_hat(V) from 2021 calibration vs 2022 empirical points.
Outputs a comparison figure in OUT_DIR.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from eelgrass_cp import softmax_rows, normalize_var

# ========================= Config =========================
CAL_BASE_DIR = r"D:\4 year training data\2021\ensemble_2020_2021_pixel_calib_cache"
CACHE_2021_NPZ = os.path.join(CAL_BASE_DIR, "pixel_calib_cache_2021.npz")
META_JSON = os.path.join(CAL_BASE_DIR, "pixel_calib_meta_2021.json")

CACHE_2022_NPZ = r"D:\2022 Data\cp_nonparametric_eval_2022\points_2022_cache_for_nonparametric.npz"

OUT_DIR = r"D:\2022 Data\cp_sigma_compare"
os.makedirs(OUT_DIR, exist_ok=True)

FIG_DPI = 220
BINS_AHAT = 20

# ========================= Load 2021 calibration =========================
print("Loading 2021 calibration cache...")
D21 = np.load(CACHE_2021_NPZ, allow_pickle=True)
with open(META_JSON, "r") as f:
    meta = json.load(f)

PLOG = D21["PLOG"]
Y21 = D21["Y"]
V21 = D21["V"]
idx_cp = D21.get("idx_cp", None)

T = float(meta["temperature_T"])
V_MIN = float(meta["variance_min"])
V_MAX = float(meta["variance_max"])
SEED = int(meta.get("seed", 42))
CAL_TEST_SPLIT = float(meta["splits"]["CAL_TEST_SPLIT"])

pool = np.arange(len(Y21)) if idx_cp is None else idx_cp

# ========================= Fit 2021 a-hat(V) =========================
P21 = softmax_rows(PLOG[pool], T)
Yp21 = Y21[pool]
Vn21 = normalize_var(V21[pool], V_MIN, V_MAX)

scores21 = 1.0 - P21[np.arange(len(P21)), Yp21]

idx_all = np.arange(len(scores21))
cal_idx, _ = train_test_split(
    idx_all, test_size=CAL_TEST_SPLIT, random_state=SEED, shuffle=True
)

df21 = pd.DataFrame({"var": Vn21[cal_idx], "score": scores21[cal_idx]})
df21["var_bin"] = pd.qcut(df21["var"], q=BINS_AHAT, duplicates="drop")

bin21 = df21.groupby("var_bin", observed=False)["score"].mean().reset_index()
centers21 = np.array([b.mid for b in bin21["var_bin"]], dtype=float)
ahat21 = bin21["score"].values.astype(float)

ord21 = np.argsort(centers21)
centers21 = centers21[ord21]
ahat21 = ahat21[ord21]

# ========================= Load 2022 points =========================
print("Loading 2022 point cache...")
D22 = np.load(CACHE_2022_NPZ)

p0 = D22["p0"]
p1 = D22["p1"]
V22 = D22["V_center"]
y22 = D22["y"]

Vn22 = normalize_var(V22, V_MIN, V_MAX)
py22 = np.where(y22 == 1, p1, p0)
scores22 = 1.0 - py22

df22 = pd.DataFrame({"var": Vn22, "score": scores22})
df22["var_bin"] = pd.qcut(df22["var"], q=BINS_AHAT, duplicates="drop")

bin22 = df22.groupby("var_bin", observed=False)["score"].mean().reset_index()
centers22 = np.array([b.mid for b in bin22["var_bin"]], dtype=float)
ahat22 = bin22["score"].values.astype(float)

ord22 = np.argsort(centers22)
centers22 = centers22[ord22]
ahat22 = ahat22[ord22]

x_max_2022 = float(np.max(centers22))

# ========================= Extrapolate 2021 curve =========================
x1_seg, x2_seg = centers21[-2], centers21[-1]
y1_seg, y2_seg = ahat21[-2], ahat21[-1]
slope_tail = (y2_seg - y1_seg) / (x2_seg - x1_seg + 1e-12)
y_ext_tail = float(y2_seg + slope_tail * (x_max_2022 - x2_seg))

x21_ext_line = np.array([centers21[-1], x_max_2022], dtype=float)
y21_ext_line = np.array([ahat21[-1], y_ext_tail], dtype=float)

# Linear reference
x0_ref, xL_ref = centers21[0], centers21[-1]
y0_ref, yL_ref = ahat21[0], ahat21[-1]
slope_ref = (yL_ref - y0_ref) / (xL_ref - x0_ref + 1e-12)
y_ref_at_xmax = float(y0_ref + slope_ref * (x_max_2022 - x0_ref))

x_ref = np.array([x0_ref, x_max_2022], dtype=float)
y_ref = np.array([y0_ref, y_ref_at_xmax], dtype=float)

# ========================= Plot =========================
plt.figure(figsize=(7.0, 4.6))

plt.plot(centers21, ahat21, "o-", lw=2, label="2021 calibration (sigma_hat)")
plt.plot(x21_ext_line, y21_ext_line, "-", lw=2, color=plt.gca().lines[-1].get_color())

plt.plot(centers22, ahat22, "s--", lw=2, label="2022 empirical")

plt.plot(x_ref, y_ref, ":", lw=2, color="gray", label="Linear reference")

plt.xlabel("Normalized ensemble variance Vn")
plt.ylabel("Estimated Scale (sigma_hat)")
plt.title("Nonparametric scale: 2021 calibration vs 2022 empirical")
plt.grid(alpha=0.35)
plt.legend()

fig_path = os.path.join(OUT_DIR, "fig_ahat_2021_vs_2022.png")
plt.tight_layout()
plt.savefig(fig_path, dpi=FIG_DPI)
plt.close()

print(f"Saved: {fig_path}")
print(f"  2021 curve extended to x_max_2022={x_max_2022:.3f} (line-only extrapolation)")
print(f"  Linear reference extended to x_max_2022={x_max_2022:.3f}")
