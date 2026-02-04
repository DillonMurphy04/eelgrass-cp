"""
Nonparametric CP evaluation on 2022 points using sigma_hat from 2021 calibration.
Compares vanilla, linear lambda=3, and nonparametric; outputs summaries and figures in OUT_DIR.
"""



import os, json, importlib, warnings
from pathlib import Path
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.interpolate import interp1d
import rasterio
from rasterio.windows import Window
import fiona
from shapely.geometry import shape
from tqdm import tqdm
from src.eelgrass_cp import qhat_conformal

# ========================= User Config =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Paths - 2021 calibration cache & meta (from  first script)
CAL_BASE_DIR  = r"D:\4 year training data\2021\ensemble_2020_2021_pixel_calib_cache"
CACHE_2021_NPZ = os.path.join(CAL_BASE_DIR, "pixel_calib_cache_2021.npz")
META_JSON      = os.path.join(CAL_BASE_DIR, "pixel_calib_meta_2021.json")

# 2022 data & output
RASTER_2022 = r"D:\2022 Data\Morro Bay Eelgrass AI - Rasterized Imagery\2022_raster_final.tif"
POINTS_SHP  = r"D:\2022 Data\2022points\2022points.shp"
OUT_DIR     = r"D:\2022 Data\cp_nonparametric_eval_2022"
os.makedirs(OUT_DIR, exist_ok=True)

CACHE_2022_NPZ = os.path.join(OUT_DIR, "points_2022_cache_for_nonparametric.npz")
RUN_2022_INFERENCE = False   # set True to rebuild 2022 cache

# Optional speed limits
MAX_POINTS_2022 = None  # None = all
CHIP_SIZE       = 448
GT_FIELD        = "GROUND_TRU"  # integer {0,1}

# Models (consistent with  pipeline)
MODEL_EMDS = {
    "unet_4yr":    r"D:\4yr_Unet_model\models\checkpoint_2025-09-19_22-47-55_epoch_11\checkpoint_2025-09-19_22-47-55_epoch_11.emd",
    "samlora_4yr": r"D:\sam_lora_model\samlora_4_year\models\checkpoint_2025-06-11_18-24-03_epoch_18\checkpoint_2025-06-11_18-24-03_epoch_18.emd",
    "samlora_2021":r"D:\sam_lora_model_2021\models\checkpoint_2025-07-30_18-52-20_epoch_18\checkpoint_2025-07-30_18-52-20_epoch_18.emd",
    "deeplab_2021":r"D:\DeepLab_2021\DeepLab_2021.emd",
    "unet_2021":   r"D:\2021 Unet\checkpoint_2025-06-25_12-39-26_epoch_11.emd",
    "deeplab_4yr": r"D:\deeplab_4_year\deeplab_4_year.emd",
}

# Figure style
FIG_DPI = 220
BINS_SIGMA = 20     # quantile bins for sigma(V)

# ========================= Utils =========================
def load_model_and_config(emd_path):
    with open(emd_path, "r") as f:
        data = json.load(f)
    model_name = data["ModelName"]
    ModelClass = getattr(importlib.import_module("arcgis.learn"), model_name)
    model_obj = ModelClass.from_model(data=None, emd_path=emd_path)
    model = model_obj.learn.model.to(DEVICE).eval()
    norm = data.get("NormalizationStats", None)
    if norm:
        cfg = {
            "min": np.array(norm["band_min_values"], np.float32),
            "max": np.array(norm["band_max_values"], np.float32),
            "scaled_mean": np.array(norm["scaled_mean_values"], np.float32),
            "scaled_std": np.array(norm["scaled_std_values"], np.float32),
        }
    else:
        cfg = {
            "min": np.array([0,0,0], np.float32),
            "max": np.array([255,255,255], np.float32),
            "scaled_mean": np.array([0.485,0.456,0.406], np.float32),
            "scaled_std": np.array([0.229,0.224,0.225], np.float32),
        }
    return model, cfg

def norm_chip(chip, cfg):
    chip_hw_c = np.transpose(chip, (1,2,0)).astype(np.float32)
    denom = np.maximum(cfg["max"] - cfg["min"], 1e-6)
    scaled = (chip_hw_c - cfg["min"]) / denom
    scaled = (scaled - cfg["scaled_mean"]) / np.maximum(cfg["scaled_std"], 1e-6)
    return np.transpose(scaled, (2,0,1))

@torch.no_grad()
def model_logits(model, chip_norm):
    x = torch.from_numpy(chip_norm).float().unsqueeze(0).to(DEVICE)
    out = model(x)
    if isinstance(out, (list,tuple)): out = out[0]
    return out.squeeze(0).detach().cpu().numpy().astype(np.float32)

def softmax_rows(logit_rows, T):
    z = logit_rows / max(T, 1e-8)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return (e / np.clip(e.sum(axis=1, keepdims=True), 1e-12, None)).astype(np.float32)

def softmax_vec(z):
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.clip(np.sum(e), 1e-12, None)

def safe_read_chip(ds, row, col, size):
    half = size // 2
    win = Window(col - half, row - half, size, size)
    chip = ds.read(window=win)  # (C,H,W)
    if chip.shape[1] < size or chip.shape[2] < size:
        return None
    return chip

def set_composition_binary_per_q(p0, p1, q, transform=None):
    """Compute empty/single/two using per-class scores transformed by 'transform' if provided."""
    s0 = 1 - p0
    s1 = 1 - p1
    if transform is not None:
        s0 = transform(s0)
        s1 = transform(s1)
    k = (s0 <= q).astype(np.int32) + (s1 <= q).astype(np.int32)
    n = len(k)
    e = int(np.sum(k == 0)); s = int(np.sum(k == 1)); t = n - e - s
    return {
        "n": n, "empty": e, "single": s, "two": t,
        "pct_empty": e/n if n else np.nan,
        "pct_single": s/n if n else np.nan,
        "pct_two": t/n if n else np.nan
    }

def corr(x, y):
    x = np.asarray(x); y = np.asarray(y)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3: return np.nan
    x0 = x[m] - x[m].mean()
    y0 = y[m] - y[m].mean()
    denom = np.sqrt((x0**2).sum()) * np.sqrt((y0**2).sum())
    return float((x0*y0).sum() / denom) if denom > 0 else np.nan

# ========================= Load 2021 cache & meta =========================
if not os.path.isfile(CACHE_2021_NPZ):
    raise FileNotFoundError(f"Missing 2021 cache: {CACHE_2021_NPZ}")
if not os.path.isfile(META_JSON):
    raise FileNotFoundError(f"Missing 2021 meta: {META_JSON}")

print(" Loading 2021 cache/meta...")
D2021 = np.load(CACHE_2021_NPZ, allow_pickle=True)
with open(META_JSON, "r") as f:
    meta = json.load(f)

ALPHA = float(meta["alpha"])
T     = float(meta["temperature_T"])
V_MIN = float(meta["variance_min"])
V_MAX = float(meta["variance_max"])
SEED  = int(meta.get("seed", 42))
CAL_TEST_SPLIT = float(meta["splits"]["CAL_TEST_SPLIT"])
CAL_TUNE_FRAC  = float(meta["splits"]["CAL_TUNE_FRAC"])

PLOG_all = D2021["PLOG"]  # [N,C]
Y_all    = D2021["Y"]     # [N]
V_all    = D2021["V"]     # [N]
idx_cp   = D2021.get("idx_cp", None)
if idx_cp is None:
    warnings.warn("idx_cp not found in cache; using all pixels as CP pool (minus any idx_temp if present).")
    pool = np.arange(len(Y_all))
else:
    pool = idx_cp

print(f"  alpha={ALPHA:.2f}  T={T:.3f}  Vmin={V_MIN:.3e}  Vmax={V_MAX:.3e}  pool_size={len(pool)}")

# Build probabilities and base scores on CP pool
P_cp = softmax_rows(PLOG_all[pool], T=T)    # [M,C]
Y_cp = Y_all[pool]
V_cp = V_all[pool]
N_pool, C = P_cp.shape

py_cp = P_cp[np.arange(N_pool), Y_cp]
base_scores_cp = 1.0 - py_cp

# Recreate cal/test split on the CP pool using saved seed
idx_all = np.arange(N_pool)
cal_idx, test_idx = train_test_split(idx_all, test_size=CAL_TEST_SPLIT, random_state=SEED, shuffle=True)
# As before, cal_sub is a proper subset of cal (used to get q)
cal_sub_idx, _ = train_test_split(cal_idx, test_size=CAL_TUNE_FRAC, random_state=SEED, shuffle=True)

# Normalize variance using *calibration-only* stats from meta (keeps consistency with  pipeline)
def normalize_var(v):
    out = (v - V_MIN) / max(V_MAX - V_MIN, 1e-8)
    return np.clip(out, 0.0, 1.0)

Vn_cp = normalize_var(V_cp)

# ========================= Fit sigma(V) via quantile binning on cal subset =========================
print(" Fitting sigma(V) from calibration (quantile bins)...")
df_cal = pd.DataFrame({"var": Vn_cp[cal_idx], "score": base_scores_cp[cal_idx]})
# 20 quantile bins by default
df_cal["var_bin"] = pd.qcut(df_cal["var"], q=BINS_SIGMA, duplicates="drop")
bin_means = df_cal.groupby("var_bin", observed=False)["score"].mean().reset_index()
bin_centers = np.array([iv.mid for iv in bin_means["var_bin"]], dtype=float)
sigma_means = bin_means["score"].values.astype(float)

# Interpolate a smooth sigma(V) function
sigma_fn = interp1d(bin_centers, sigma_means, kind="linear", fill_value="extrapolate", assume_sorted=False)

# Plot sigma(V) curve
plt.figure(figsize=(6.8,4.2))
plt.plot(bin_centers, sigma_means, "o-", lw=2)
plt.xlabel("Linear ensemble variance Vn")
plt.ylabel("Estimated scale a(V) = E[1 - p_y | Vn]")
plt.title("Nonparametric scale from 2021 calibration (quantile-binned)")
plt.grid(alpha=0.35)
fig_sigma = os.path.join(OUT_DIR, "fig_sigma_curve.png")
plt.tight_layout(); plt.savefig(fig_sigma, dpi=FIG_DPI); plt.close()
print(f"  -  {fig_sigma}")

# Calibrate q on nonparametric scores (use cal_sub)
s_cal = base_scores_cp[cal_idx]
vn_cal = Vn_cp[cal_idx]
s_nonparam_cal = s_cal / sigma_fn(vn_cal)

# Use cal_sub_idx *mapped into cal_idx space* for q (mirror  pipeline)
# Build a mask to pick cal_sub entries
mask_cal = np.zeros(len(cal_idx), dtype=bool)
# cal_idx indices are w.r.t. idx_all; cal_sub_idx also is; we need positions within cal_idx
pos_in_cal = {g:i for i,g in enumerate(cal_idx)}
for g in cal_sub_idx:
    if g in pos_in_cal:
        mask_cal[pos_in_cal[g]] = True
qhat_nonparam = qhat_conformal(s_nonparam_cal[mask_cal], ALPHA)
print(f"  qhat_nonparametric (from 2021 cal_sub) = {qhat_nonparam:.6f}")

# Also pull vanilla and (optionally) linear lambda=3 thresholds from meta for comparison
qhat_vanilla = float(meta["pixel_level"]["vanilla"]["qhat"])
qhat_linear_l3 = None
adaptive = meta.get("pixel_level", {}).get("adaptive", {})
linear_pack = adaptive.get("linear", adaptive.get("normalized", {}))
if linear_pack:
    for lam_str, pack in linear_pack.items():
        if abs(float(lam_str) - 3.0) < 1e-8:
            qhat_linear_l3 = float(pack["qhat"])
            break
print(f"Lambda = 3 qhat: {qhat_linear_l3}")
# ========================= Load or build 2022 per-point cache =========================
def maybe_run_or_load_points_2022():
    if (not RUN_2022_INFERENCE) and os.path.isfile(CACHE_2022_NPZ):
        print(f"Loading 2022 cache: {CACHE_2022_NPZ}")
        D = np.load(CACHE_2022_NPZ)
        return D["p0"], D["p1"], D["V_center"], D["y"], D["ids"]
    print(" Running ensemble on 2022 points to build cache...")
    # Load models
    ensemble = {}
    for key, emd in MODEL_EMDS.items():
        try:
            m, cfg = load_model_and_config(emd)
            ensemble[key] = {"model": m, "config": cfg}
            print(f"  -  Loaded {key}")
        except Exception as e:
            warnings.warn(f"Failed to load {key}: {e}")
    if not ensemble:
        raise RuntimeError("No models for 2022 inference")

    # Read points
    with fiona.open(POINTS_SHP, "r") as src:
        feats = [(shape(f["geometry"]), f["properties"]) for f in src]
    if MAX_POINTS_2022 is not None:
        feats = feats[:MAX_POINTS_2022]
    print(f"  -  Points: {len(feats)}")

    p0_list, p1_list, v_list, y_list, id_list = [], [], [], [], []
    with rasterio.open(RASTER_2022) as ds:
        for i, (geom, props) in enumerate(tqdm(feats, desc="Points2022")):
            x, y_pt = geom.x, geom.y
            row, col = ds.index(x, y_pt)
            chip = safe_read_chip(ds, row, col, CHIP_SIZE)
            if chip is None:
                continue
            per_model_logits = []
            per_model_probs_c1 = []
            for key, pack in ensemble.items():
                try:
                    chip_norm = norm_chip(chip, pack["config"])
                    logits = model_logits(pack["model"], chip_norm)  # [C,H,W]
                    per_model_logits.append(logits)
                    z = logits - logits.max(axis=0, keepdims=True)
                    e = np.exp(z); probs = e / np.clip(e.sum(axis=0, keepdims=True), 1e-12, None)
                    per_model_probs_c1.append(probs[min(1, probs.shape[0]-1)])
                except Exception as e:
                    warnings.warn(f"Inference failed for model {key} at point {i}: {e}")
            if len(per_model_logits) == 0:
                continue
            L_stack = np.stack(per_model_logits, axis=0)
            mean_logits = np.mean(L_stack, axis=0)
            var_map = np.var(np.stack(per_model_probs_c1, axis=0), axis=0)
            cy, cx = CHIP_SIZE//2, CHIP_SIZE//2
            logits_center = mean_logits[:, cy, cx]
            probs_center  = softmax_vec(logits_center / max(T, 1e-8))
            y_true = props.get(GT_FIELD, None)
            if y_true is None:
                continue
            p0_list.append(float(probs_center[0]))
            p1_list.append(float(probs_center[min(1,len(probs_center)-1)]))
            v_list.append(float(var_map[cy, cx]))
            y_list.append(int(y_true))
            id_list.append(i)

    p0 = np.array(p0_list, dtype=np.float32)
    p1 = np.array(p1_list, dtype=np.float32)
    Vc = np.array(v_list,   dtype=np.float32)
    y  = np.array(y_list,   dtype=np.int16)
    ids= np.array(id_list,  dtype=np.int64)
    np.savez_compressed(CACHE_2022_NPZ, p0=p0, p1=p1, V_center=Vc, y=y, ids=ids)
    print(f" Saved 2022 cache: {CACHE_2022_NPZ} (N={len(ids)})")
    return p0, p1, Vc, y, ids

print(" Loading 2022 points...")
p0, p1, Vc, y2022, ids2022 = maybe_run_or_load_points_2022()
N2022 = len(ids2022)
print(f"  N2022 = {N2022}")

# 2022 derived arrays
Vn2022 = normalize_var(Vc)
py2022 = np.where(y2022 == 1, p1, p0)
base_scores_2022 = 1.0 - py2022

# Helper: linear-lambda3 scores (if available) (changed to lambda = 3)
def linear_lambda3_scores(scores_base, vnorm):
    # s' = s / (1 + 3 * Vn)
    return scores_base / (1.0 + 3.0 * vnorm)

# Nonparametric scores
sigma_hat_2022 = sigma_fn(Vn2022)
s_nonparam_2022  = base_scores_2022 / sigma_hat_2022

# ========================= Evaluate methods on 2022 =========================
def eval_method(scores, q, y, p0, p1, transform_for_sets=None):
    covered = (scores <= q)
    cov = float(np.mean(covered)) if len(covered) else float("nan")
    # per-class coverage
    pc = {}
    for c in [0,1]:
        idx = (y == c)
        pc[c] = float(np.mean(scores[idx] <= q)) if np.any(idx) else np.nan
    # set composition
    sc = set_composition_binary_per_q(p0, p1, q, transform=transform_for_sets)
    return cov, pc, sc

# Vanilla
cov_van, pc_van, sc_van = eval_method(
    base_scores_2022, qhat_vanilla, y2022, p0, p1, transform_for_sets=None
)

# Linear lambda=3 (if q available)
cov_linear3 = pc_linear3 = sc_linear3 = None
if qhat_linear_l3 is not None:
    s_linear3_2022 = linear_lambda3_scores(base_scores_2022, Vn2022)
    cov_linear3, pc_linear3, sc_linear3 = eval_method(
        s_linear3_2022, qhat_linear_l3, y2022, p0, p1,
        transform_for_sets=(lambda s: s / (1.0 + 3.0 * Vn2022))
    )

# Nonparametric CP
cov_nonparam, pc_nonparam, sc_nonparam = eval_method(
    s_nonparam_2022, qhat_nonparam, y2022, p0, p1,
    transform_for_sets=(lambda s: s / sigma_hat_2022)
)

# Conditional diagnostics: coverage vs variance correlation (bin-wise)
def binned_coverage(scores, q, y, vnorm, nbins=12):
    # bin by variance, compute coverage per bin
    edges = np.quantile(vnorm, np.linspace(0,1,nbins+1))
    edges[0], edges[-1] = 0.0, 1.0
    covs, mids, counts = [], [], []
    for i in range(nbins):
        m = (vnorm >= edges[i]) & (vnorm < edges[i+1]) if i < nbins-1 else (vnorm >= edges[i]) & (vnorm <= edges[i+1])
        if m.sum() == 0:
            covs.append(np.nan); mids.append(0.5*(edges[i]+edges[i+1])); counts.append(0)
        else:
            covs.append(float(np.mean(scores[m] <= q)))
            mids.append(0.5*(edges[i]+edges[i+1]))
            counts.append(int(m.sum()))
    return np.array(mids), np.array(covs), np.array(counts)

m_van, covs_van, cnt_van = binned_coverage(base_scores_2022, qhat_vanilla, y2022, Vn2022, nbins=12)
if qhat_linear_l3 is not None:
    s_linear3_2022 = linear_lambda3_scores(base_scores_2022, Vn2022)
    m_linear, covs_linear, cnt_linear = binned_coverage(s_linear3_2022, qhat_linear_l3, y2022, Vn2022, nbins=12)
m_nonparam, covs_nonparam, cnt_nonparam = binned_coverage(s_nonparam_2022, qhat_nonparam, y2022, Vn2022, nbins=12)

# Correlations (using bin means where counts>0)
def cov_var_corr(bin_mids, bin_covs, counts):
    m = np.isfinite(bin_covs) & (counts > 0)
    return corr(bin_mids[m], bin_covs[m])

r_van  = cov_var_corr(m_van,  covs_van,  cnt_van)
r_linear = cov_var_corr(m_linear, covs_linear, cnt_linear) if qhat_linear_l3 is not None else None
r_nonparam  = cov_var_corr(m_nonparam,  covs_nonparam,  cnt_nonparam)

# ========================= Evaluate methods on 2021 (ID test subset) =========================
# Use the CP pool test subset as the in-distribution test set
P_test_2021   = P_cp[test_idx]
Y_test_2021   = Y_cp[test_idx]
Vn_test_2021  = Vn_cp[test_idx]
base_scores_2021 = base_scores_cp[test_idx]

# per-class probs for 2021 test subset
p0_2021 = P_test_2021[:, 0]
p1_2021 = P_test_2021[:, 1]

# nonparametric scores on 2021
sigma_hat_2021 = sigma_fn(Vn_test_2021)
s_nonparam_2021  = base_scores_2021 / sigma_hat_2021

# vanilla (ID)
cov_van_2021, pc_van_2021, sc_van_2021 = eval_method(
    base_scores_2021, qhat_vanilla, Y_test_2021, p0_2021, p1_2021, transform_for_sets=None
)

# nonparametric (nonparametric sigma(V)) on ID
cov_nonparam_2021, pc_nonparam_2021, sc_nonparam_2021 = eval_method(
    s_nonparam_2021, qhat_nonparam, Y_test_2021, p0_2021, p1_2021,
    transform_for_sets=(lambda s: s / sigma_hat_2021)
)

# ========================= Save per-point CSV =========================
df = pd.DataFrame({
    "id": ids2022, "gt": y2022,
    "p0": p0, "p1": p1, "V": Vc, "Vn": Vn2022,
    "base_score": base_scores_2022,
    "covered_vanilla": (base_scores_2022 <= qhat_vanilla).astype(np.int32),
    "score_nonparametric": s_nonparam_2022,
    "covered_nonparametric": (s_nonparam_2022 <= qhat_nonparam).astype(np.int32),
})
if qhat_linear_l3 is not None:
    s_linear3_2022 = linear_lambda3_scores(base_scores_2022, Vn2022)
    df["score_linear_l3"] = s_linear3_2022
    df["covered_linear_l3"] = (s_linear3_2022 <= qhat_linear_l3).astype(np.int32)

csv_points = os.path.join(OUT_DIR, "per_point_2022_with_nonparametric.csv")
df.to_csv(csv_points, index=False)
print(f" Saved per-point CSV: {csv_points}")

# ========================= Save summary JSON =========================
summary = {
    "alpha": ALPHA,
    "temperature_T": T,
    "variance_norm": {"min": V_MIN, "max": V_MAX},
    "counts": {
        "N_2022": int(N2022),
        "n_class0": int(np.sum(y2022==0)),
        "n_class1": int(np.sum(y2022==1)),
    },
    "methods": {
        "vanilla": {
            "qhat": qhat_vanilla,
            "coverage": cov_van,
            "per_class_coverage": {int(k): float(v) for k,v in pc_van.items()},
            "set_composition": {k: float(v) if isinstance(v,(int,float)) else v for k,v in sc_van.items()},
            "cov_vs_var_corr": r_van
        },
        "nonparametric": {
            "qhat": qhat_nonparam,
            "coverage": cov_nonparam,
            "per_class_coverage": {int(k): float(v) for k,v in pc_nonparam.items()},
            "set_composition": {k: float(v) if isinstance(v,(int,float)) else v for k,v in sc_nonparam.items()},
            "cov_vs_var_corr": r_nonparam,
            "bins_used_for_sigma": int(BINS_SIGMA)
        }
    },
    "files": {
        "per_point_csv": Path(csv_points).name
    }
}
if qhat_linear_l3 is not None:
    summary["methods"]["linear_lambda3"] = {
        "qhat": qhat_linear_l3,
        "coverage": cov_linear3,
        "per_class_coverage": {int(k): float(v) for k,v in pc_linear3.items()},
        "set_composition": {k: float(v) if isinstance(v,(int,float)) else v for k,v in sc_linear3.items()},
        "cov_vs_var_corr": r_linear
    }

summary_json = os.path.join(OUT_DIR, "summary_nonparametric_2022.json")
with open(summary_json, "w") as f:
    json.dump(summary, f, indent=2)
print(f" Saved summary JSON: {summary_json}")

# ========================= Figures =========================
print(" Creating figures...")

# 1) sigma(V) curve already saved as fig_sigma_curve.png (above)

# 2) Coverage vs variance bins (Vanilla vs Linear lambda=3 vs Nonparametric)
plt.figure(figsize=(7.8,4.8))
plt.plot(m_van, covs_van, "o-", lw=2, label="Vanilla")
if qhat_linear_l3 is not None:
    plt.plot(m_linear, covs_linear, "s--", lw=2, label="Linear lambda=3")
    plt.plot(m_nonparam, covs_nonparam, "d-.", lw=2, label="Nonparametric")
plt.axhline(1-ALPHA, color="k", ls=":", lw=1.2, alpha=0.8)
plt.xlabel("Linear variance (bin mid)")
plt.ylabel("Coverage (per bin)")
plt.title("Coverage vs Variance - 2022")
plt.grid(alpha=0.35); plt.legend()
fig_cov_vs_var = os.path.join(OUT_DIR, "fig_coverage_vs_variance_bins.png")
plt.tight_layout(); plt.savefig(fig_cov_vs_var, dpi=FIG_DPI); plt.close()
print(f"  -  {fig_cov_vs_var}")

# ============================================================
# 2b) Conditional per-class coverage vs variance (new)
# ============================================================

def binned_coverage_per_class(scores, q, y, vnorm, nbins=12):
    """Compute per-class coverage vs variance using equal-width bins."""
    edges = np.linspace(0, 1, nbins + 1)
    mids = 0.5 * (edges[:-1] + edges[1:])
    covs_c0, covs_c1 = [], []
    for i in range(nbins):
        m_bin = (vnorm >= edges[i]) & (vnorm < edges[i+1]) if i < nbins-1 else (vnorm >= edges[i]) & (vnorm <= edges[i+1])
        for c, store in zip([0, 1], [covs_c0, covs_c1]):
            m = m_bin & (y == c)
            if np.any(m):
                store.append(float(np.mean(scores[m] <= q)))
            else:
                store.append(np.nan)
    return mids, np.array(covs_c0), np.array(covs_c1)

# Compute for each method
m_van_c, c0_van, c1_van = binned_coverage_per_class(base_scores_2022, qhat_vanilla, y2022, Vn2022)
if qhat_linear_l3 is not None:
    s_linear3_2022 = linear_lambda3_scores(base_scores_2022, Vn2022)
    m_linear_c, c0_linear, c1_linear = binned_coverage_per_class(s_linear3_2022, qhat_linear_l3, y2022, Vn2022)
m_nonparam_c, c0_nonparam, c1_nonparam = binned_coverage_per_class(s_nonparam_2022, qhat_nonparam, y2022, Vn2022)

# Plot conditional coverage per class
fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
for ax, cls, cov0, cov1 in zip(axes, [0, 1], [c0_van, c1_van], [c0_van, c1_van]):
    if cls == 0:
        ax.plot(m_van_c, c0_van, "o-", lw=2, label="Vanilla")
        if qhat_linear_l3 is not None:
            ax.plot(m_linear_c, c0_linear, "s--", lw=2, label="Linear lambda=3")
            ax.plot(m_nonparam_c, c0_nonparam, "d-.", lw=2, label="Nonparametric")
    else:
        ax.plot(m_van_c, c1_van, "o-", lw=2, label="Vanilla")
        if qhat_linear_l3 is not None:
            ax.plot(m_linear_c, c1_linear, "s--", lw=2, label="Linear lambda=3")
            ax.plot(m_nonparam_c, c1_nonparam, "d-.", lw=2, label="Nonparametric")
    ax.axhline(1-ALPHA, color="k", ls=":", lw=1.1, alpha=0.8)
    ax.set_xlabel("Linear variance (bin mid)")
    ax.set_title(f"Class {cls} coverage vs variance")
    ax.grid(alpha=0.35)
axes[0].set_ylabel("Coverage (per bin)")
axes[1].legend(loc="lower right", fontsize=8)
fig_cond_cov = os.path.join(OUT_DIR, "fig_conditional_per_class_coverage_vs_variance.png")
plt.tight_layout(); plt.savefig(fig_cond_cov, dpi=FIG_DPI); plt.close()
print(f"  -  {fig_cond_cov}")


# 3) Scatter: Vanilla base score vs Nonparametric score (with q lines)
plt.figure(figsize=(6.1,6.1))
x = base_scores_2022
y_sc = s_nonparam_2022
plt.scatter(x, y_sc, s=8, alpha=0.35)
# q lines
plt.axvline(qhat_vanilla, color='tab:blue',  ls='--', lw=1.6, label='q Vanilla')
plt.axhline(qhat_nonparam,  color='tab:orange',ls='--', lw=1.6, label='q Nonparametric')
# diagonal
lo = min(float(np.nanmin(x)), float(np.nanmin(y_sc)))
hi = max(float(np.nanmax(x)), float(np.nanmax(y_sc)))
plt.plot([lo,hi],[lo,hi], color='grey', ls=':', lw=1.0, alpha=0.8)
plt.xlim(lo,hi); plt.ylim(lo,hi)
plt.xlabel("Vanilla base score  (s = 1 - p_y)")
plt.ylabel("Nonparametric score (s / sigma(V))")
plt.title("Per-point Score Transformation: Vanilla -> Nonparametric")
plt.grid(alpha=0.3); plt.legend()
fig_scatter = os.path.join(OUT_DIR, "fig_scatter_vanilla_vs_nonparametric.png")
plt.tight_layout(); plt.savefig(fig_scatter, dpi=FIG_DPI); plt.close()
print(f"  -  {fig_scatter}")

# 4) Coverage vs two-label % (Nonparametric only - analogous to  lambda plots)
two_pct_nonparam = sc_nonparam["pct_two"] * 100.0
plt.figure(figsize=(6.6,4.4))
plt.scatter([two_pct_nonparam], [cov_nonparam], s=60)
plt.axhline(1-ALPHA, color='k', ls=':', lw=1.1)
plt.xlabel("Two-label set percentage (%)")
plt.ylabel("Coverage")
plt.title("Nonparametric CP: Coverage vs Two-label Frequency (2022)")
plt.grid(alpha=0.35)
fig_cov_two = os.path.join(OUT_DIR, "fig_nonparametric_coverage_vs_two_label.png")
plt.tight_layout(); plt.savefig(fig_cov_two, dpi=FIG_DPI); plt.close()
print(f"  -  {fig_cov_two}")

# ========================= Console report =========================
def pct(x): return f"{x*100:.1f}%" if isinstance(x,(int,float)) and np.isfinite(x) else "nan"
print("\n=== 2021 In-distribution Results (CP test subset) ===")
print(
    f"Vanilla:       q={qhat_vanilla:.4f}  cov={cov_van_2021:.3f}  "
    f"single={pct(sc_van_2021['pct_single'])}  empty={pct(sc_van_2021['pct_empty'])}  "
    f"two={pct(sc_van_2021['pct_two'])}"
)
print(
    f"Nonparametric: q={qhat_nonparam:.4f}  cov={cov_nonparam_2021:.3f}  "
    f"single={pct(sc_nonparam_2021['pct_single'])}  empty={pct(sc_nonparam_2021['pct_empty'])}  "
    f"two={pct(sc_nonparam_2021['pct_two'])}"
)
print("\n=== 2022 Results ===")
print(f"Vanilla:         q={qhat_vanilla:.5f} | cov={cov_van:.3f} | set: single={pct(sc_van['pct_single'])} "
      f"empty={pct(sc_van['pct_empty'])} two={pct(sc_van['pct_two'])} | cov-var corr={r_van:+.3f}")
if qhat_linear_l3 is not None:
    print(f"Linear lambda=3:  q={qhat_linear_l3:.5f} | cov={cov_linear3:.3f} | set: single={pct(sc_linear3['pct_single'])} "
          f"empty={pct(sc_linear3['pct_empty'])} two={pct(sc_linear3['pct_two'])} | cov-var corr={r_linear:+.3f}")
print(f"Nonparametric: q={qhat_nonparam:.5f} | cov={cov_nonparam:.3f} | set: single={pct(sc_nonparam['pct_single'])} "
      f"empty={pct(sc_nonparam['pct_empty'])} two={pct(sc_nonparam['pct_two'])} | cov-var corr={r_nonparam:+.3f}")

# ============================================================
# Print class-conditional coverage summary
# ============================================================

print("\n=== Conditional Coverage (per class) ===")
header = f"{'Method':<20}{'Class0':>10}{'Class1':>10}{'Overall':>10}"
print(header)
print("-" * len(header))

def fmt(x): 
    return f"{x:>10.3f}" if x is not None and np.isfinite(x) else f"{'nan':>10}"

rows = [
    ("Vanilla", pc_van[0], pc_van[1], cov_van),
]
if qhat_linear_l3 is not None:
    rows.append(("Linear lambda=3", pc_linear3[0], pc_linear3[1], cov_linear3))
rows.append(("Nonparametric CP", pc_nonparam[0], pc_nonparam[1], cov_nonparam))

for name, c0, c1, cov in rows:
    print(f"{name:<20}{fmt(c0)}{fmt(c1)}{fmt(cov)}")


print("\n Outputs:")
print("  CSV :", csv_points)
print("  JSON:", summary_json)
print("  FIGS:", Path(fig_sigma).name, Path(fig_cov_vs_var).name, Path(fig_scatter).name, Path(fig_cov_two).name)
