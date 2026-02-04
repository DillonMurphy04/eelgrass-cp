"""
Builds the 2021 pixel-level calibration cache and meta for CP.
Outputs: pixel_calib_cache_2021.npz and pixel_calib_meta_2021.json.
"""

import os
import json
import warnings
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from src.eelgrass_cp import load_model_and_config, norm_chip, model_logits
from src.eelgrass_cp import softmax_rows
from src.eelgrass_cp import TemperatureScaler, fit_temperature, stratified_indices
from src.eelgrass_cp import qhat_split_conformal, set_composition_binary
from src.eelgrass_cp import per_class_coverage, normalize_var

# ========================= User Config =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = r"D:\4 year training data\2021"
MAP_TXT = os.path.join(BASE_DIR, "map.txt")
OUT_DIR = os.path.join(BASE_DIR, "ensemble_2020_2021_pixel_calib_cache")
os.makedirs(OUT_DIR, exist_ok=True)

RUN_INFERENCE = False

CACHE_NPZ = os.path.join(OUT_DIR, "pixel_calib_cache_2021.npz")
META_JSON = os.path.join(OUT_DIR, "pixel_calib_meta_2021.json")

MAX_TILES = 1500
TEMP_SAMPLE_PIX = 800_000
CP_SAMPLE_PIX = 2_000_000

ALPHA = 0.10
CAL_TEST_SPLIT = 0.50
CAL_TUNE_FRAC = 0.30
LAMBDA_GRID = [0.5, 1.0, 2.0, 3.0, 4.0]

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

MODEL_EMDS = {
    "unet_4yr": r"D:\4yr_Unet_model\models\checkpoint_2025-09-19_22-47-55_epoch_11\checkpoint_2025-09-19_22-47-55_epoch_11.emd",
    "samlora_4yr": r"D:\sam_lora_model\samlora_4_year\models\checkpoint_2025-06-11_18-24-03_epoch_18\checkpoint_2025-06-11_18-24-03_epoch_18.emd",
    "samlora_2021": r"D:\sam_lora_model_2021\models\checkpoint_2025-07-30_18-52-20_epoch_18\checkpoint_2025-07-30_18-52-20_epoch_18.emd",
    "deeplab_2021": r"D:\DeepLab_2021\DeepLab_2021.emd",
    "unet_2021": r"D:\2021 Unet\checkpoint_2025-06-25_12-39-26_epoch_11.emd",
    "deeplab_4yr": r"D:\deeplab_4_year\deeplab_4_year.emd",
}


# ========================= Cache / Inference =========================

def maybe_run_inference_or_load_cache():
    if (not RUN_INFERENCE) and os.path.isfile(CACHE_NPZ):
        print(f"Loading cache: {CACHE_NPZ}")
        data = np.load(CACHE_NPZ, allow_pickle=True)
        return data["PLOG"], data["Y"], data["V"]

    print("Running ensemble inference on 2021 to build cache...")
    ensemble = {}
    for key, emd in MODEL_EMDS.items():
        try:
            m, cfg = load_model_and_config(emd, device=DEVICE)
            ensemble[key] = {"model": m, "config": cfg}
            print(f"  Loaded {key}")
        except Exception as e:
            warnings.warn(f"Failed to load {key}: {e}")
    if not ensemble:
        raise RuntimeError("No models loaded")

    pairs = []
    with open(MAP_TXT, "r") as f:
        for line in f:
            if not line.strip():
                continue
            img_rel, lbl_rel = line.strip().split()
            pairs.append((os.path.join(BASE_DIR, img_rel), os.path.join(BASE_DIR, lbl_rel)))
    print(f"Found {len(pairs)} image/label pairs.")
    if len(pairs) > MAX_TILES:
        pairs = random.sample(pairs, MAX_TILES)
    print(f"Using {len(pairs)} tiles.")

    pixel_logits = []
    pixel_labels = []
    pixel_var = []

    for img_path, lbl_path in tqdm(pairs, desc="Tiles"):
        img = np.array(Image.open(img_path).convert("RGB")).astype(np.float32)
        gt = np.array(Image.open(lbl_path)).astype(np.int32)
        chip = np.moveaxis(img, -1, 0)

        per_model_logits = []
        per_model_probs_c1 = []
        for key, pack in ensemble.items():
            chip_norm = norm_chip(chip, pack["config"])
            logits = model_logits(pack["model"], chip_norm, device=DEVICE)
            per_model_logits.append(logits)

            z = logits - logits.max(axis=0, keepdims=True)
            e = np.exp(z)
            probs = e / np.clip(e.sum(axis=0, keepdims=True), 1e-12, None)
            per_model_probs_c1.append(probs[min(1, probs.shape[0] - 1)])

        L_stack = np.stack(per_model_logits, axis=0)
        mean_logits = np.mean(L_stack, axis=0)
        var_map = np.var(np.stack(per_model_probs_c1, axis=0), axis=0)

        C, H, W = mean_logits.shape
        pixel_logits.append(mean_logits.reshape(C, -1).T)
        pixel_labels.append(gt.reshape(-1))
        pixel_var.append(var_map.reshape(-1))

    PLOG_all = np.concatenate(pixel_logits, axis=0)
    Y_all = np.concatenate(pixel_labels, axis=0)
    V_all = np.concatenate(pixel_var, axis=0)

    mask = (Y_all >= 0)
    PLOG_all, Y_all, V_all = PLOG_all[mask], Y_all[mask], V_all[mask]

    np.savez_compressed(
        CACHE_NPZ,
        PLOG=PLOG_all.astype(np.float32),
        Y=Y_all.astype(np.int16),
        V=V_all.astype(np.float32),
        alpha=float(ALPHA),
        lambda_grid=np.array(LAMBDA_GRID, dtype=np.float32),
        seed=np.int32(SEED),
    )
    print(f"Saved cache: {CACHE_NPZ}  (N={len(Y_all)})")
    return PLOG_all, Y_all, V_all


PLOG_all, Y_all, V_all = maybe_run_inference_or_load_cache()
N_all, C = PLOG_all.shape
print(f"Cache ready: PLOG[{N_all},{C}], labels[{len(Y_all)}], variance[{len(V_all)}]")

# Temperature scaling
n_temp = min(TEMP_SAMPLE_PIX, N_all)
idx_temp = stratified_indices(Y_all, n_temp, seed=SEED)
if len(idx_temp) == 0:
    idx_temp = np.random.choice(np.arange(N_all), size=n_temp, replace=False)
T_learned = fit_temperature(PLOG_all[idx_temp], Y_all[idx_temp], device=DEVICE)
print(f"Learned temperature T = {T_learned:.3f}")

# CP pool disjoint from temp fit
pool = np.setdiff1d(np.arange(N_all), idx_temp, assume_unique=False)
n_cp = min(CP_SAMPLE_PIX, len(pool))
idx_cp = np.random.choice(pool, size=n_cp, replace=False)

PLOG = PLOG_all[idx_cp]
Y = Y_all[idx_cp]
V = V_all[idx_cp]
N = len(Y)

# Update cache with T and indices
try:
    data = np.load(CACHE_NPZ, allow_pickle=True)
    np.savez_compressed(
        CACHE_NPZ,
        PLOG=data["PLOG"], Y=data["Y"], V=data["V"],
        alpha=float(ALPHA),
        lambda_grid=np.array(LAMBDA_GRID, dtype=np.float32),
        seed=np.int32(SEED),
        T=float(T_learned),
        idx_temp=idx_temp.astype(np.int64),
        idx_cp=idx_cp.astype(np.int64),
    )
    print("Updated cache with T and index bookkeeping.")
except Exception as e:
    warnings.warn(f"Could not update cache with T/indices: {e}")

# Build probabilities on CP pool
P = softmax_rows(PLOG, T=T_learned)
py = P[np.arange(N), Y]
base_scores = 1.0 - py

idx_all = np.arange(N)
cal_idx, test_idx = train_test_split(idx_all, test_size=CAL_TEST_SPLIT, random_state=SEED, shuffle=True)
cal_sub_idx, _ = train_test_split(cal_idx, test_size=CAL_TUNE_FRAC, random_state=SEED, shuffle=True)

# Variance normalization using calibration-only stats
Vmin = float(np.min(V[cal_idx])) if len(cal_idx) else float(np.min(V))
Vmax = float(np.max(V[cal_idx])) if len(cal_idx) else float(np.max(V))
Vn = normalize_var(V, Vmin, Vmax)

qhat_van = qhat_split_conformal(base_scores, ALPHA, cal_sub_idx)
cov_van = float(np.mean(base_scores[test_idx] <= qhat_van))
setcomp_van_test = set_composition_binary(P[test_idx, 0], P[test_idx, 1], qhat_van)


def eval_rule_all_lambdas(rule_name, lambdas):
    def score_transform_linear(base, vnorm, lam):
        return base / (1.0 + lam * vnorm)

    out = {}
    for lam in lambdas:
        s = score_transform_linear(base_scores, Vn, lam)
        q = qhat_split_conformal(s, ALPHA, cal_sub_idx)
        cov = float(np.mean(s[test_idx] <= q)) if len(test_idx) else float("nan")
        pc = per_class_coverage(s, Y, q)
        sc = set_composition_binary(P[test_idx, 0], P[test_idx, 1], q,
                                    transform=(lambda s0: score_transform_linear(s0, Vn[test_idx], lam)))
        out[str(lam)] = {
            "lambda": float(lam),
            "qhat": float(q),
            "coverage_test": float(cov),
            "coverage_test_per_class": pc,
            "setcomp_test": sc,
        }
    return out


adaptive_results = {
    "linear": eval_rule_all_lambdas("linear", LAMBDA_GRID),
}

meta = {
    "alpha": float(ALPHA),
    "temperature_T": float(T_learned),
    "variance_min": float(Vmin),
    "variance_max": float(Vmax),
    "lambda_grid": list(map(float, LAMBDA_GRID)),
    "splits": {
        "N_all_pixels_cached": int(N_all),
        "N_cp_pool": int(N),
        "CAL_TEST_SPLIT": float(CAL_TEST_SPLIT),
        "CAL_TUNE_FRAC": float(CAL_TUNE_FRAC),
        "sizes": {
            "n_temp_fit": int(len(idx_temp)),
            "n_cal": int(len(cal_idx)),
            "n_cal_proper": int(len(cal_sub_idx)),
            "n_test": int(len(test_idx)),
        },
    },
    "pixel_level": {
        "vanilla": {
            "qhat": float(qhat_van),
            "coverage_test": float(cov_van),
            "setcomp_test": {
                "n": int(setcomp_van_test["n"]),
                "empty": int(setcomp_van_test["empty"]),
                "single": int(setcomp_van_test["single"]),
                "two": int(setcomp_van_test["two"]),
                "pct_empty": float(setcomp_van_test["pct_empty"]),
                "pct_single": float(setcomp_van_test["pct_single"]),
                "pct_two": float(setcomp_van_test["pct_two"]),
            },
        },
        "adaptive": adaptive_results,
    },
    "files": {
        "cache_npz": Path(CACHE_NPZ).name,
    },
    "notes": {
        "scores_base": "base = 1 - p_y after temperature",
        "variance_norm_stats": "min/max from calibration subset only; applied to both cal and test.",
        "adaptive_forms": {
            "linear": "s = base / (1 + lambda Vn)",
        },
    },
}

with open(META_JSON, "w") as f:
    json.dump(meta, f, indent=2)
print(f"Saved meta JSON: {META_JSON}")


def pct(x):
    return f"{x*100:.1f}%"

print("\n=== Summary (CP test split; disjoint from temp fit) ===")
print(f"T = {T_learned:.3f} | alpha = {ALPHA} | Vmin={Vmin:.3e} Vmax={Vmax:.3e} (from cal subset)")
sc_v = setcomp_van_test
print(f"Vanilla:    qhat={qhat_van:.4f}  cov={cov_van:.3f}  "
      f"single={pct(sc_v['pct_single'])}  empty={pct(sc_v['pct_empty'])}  two={pct(sc_v['pct_two'])}")
for rule_name, res in adaptive_results.items():
    for lam in LAMBDA_GRID:
        r = res[str(lam)]
        sc = r["setcomp_test"]
        print(f"{rule_name:14s} lambda={lam:.2f}  qhat={r['qhat']:.4f}  cov={r['coverage_test']:.3f}  "
              f"single={pct(sc['pct_single'])}  empty={pct(sc['pct_empty'])}  two={pct(sc['pct_two'])}")
