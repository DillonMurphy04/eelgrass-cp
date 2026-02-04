"""
Compare linear CP across lambda values on 2022 points.
Inputs: 2021 calibration meta + 2022 points cache (or run inference).
Outputs: per-point CSV, summary JSON, and figures in OUT_DIR.
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
from tqdm import tqdm
import matplotlib.pyplot as plt

from eelgrass_cp import load_model_and_config, norm_chip, model_logits
from eelgrass_cp import softmax_vec
from eelgrass_cp import safe_read_chip
from eelgrass_cp import set_composition_binary, per_class_coverage

# ========================= User Config =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RASTER_PATH = r"D:\2022 Data\Morro Bay Eelgrass AI - Rasterized Imagery\2022_raster_final.tif"
POINTS_SHP = r"D:\2022 Data\2022points\2022points.shp"
OUT_DIR = r"D:\2022 Data\cp_eval_2022_points_compare"
os.makedirs(OUT_DIR, exist_ok=True)

CACHE_2022_NPZ = os.path.join(OUT_DIR, "points_2022_cache.npz")
RUN_2022_INFERENCE = False

META_JSON = r"D:\4 year training data\2021\ensemble_2020_2021_pixel_calib_cache\pixel_calib_meta_2021.json"

MAX_POINTS = None

CHIP_SIZE = 448
GT_FIELD = "GROUND_TRU"
N_CLASSES = 2

COMPARE_LAMBDAS = [0.5, 1.0, 2.0, 3.0, 4.0]

MODEL_EMDS = {
    "unet_4yr": r"D:\4yr_Unet_model\models\checkpoint_2025-09-19_22-47-55_epoch_11\checkpoint_2025-09-19_22-47-55_epoch_11.emd",
    "samlora_4yr": r"D:\sam_lora_model\samlora_4_year\models\checkpoint_2025-06-11_18-24-03_epoch_18\checkpoint_2025-06-11_18-24-03_epoch_18.emd",
    "samlora_2021": r"D:\sam_lora_model_2021\models\checkpoint_2025-07-30_18-52-20_epoch_18\checkpoint_2025-07-30_18-52-20_epoch_18.emd",
    "deeplab_2021": r"D:\DeepLab_2021\DeepLab_2021.emd",
    "unet_2021": r"D:\2021 Unet\checkpoint_2025-06-25_12-39-26_epoch_11.emd",
    "deeplab_4yr": r"D:\deeplab_4_year\deeplab_4_year.emd",
}

# ========================= Load Calibration Meta =========================
print("Loading calibration meta...")
with open(META_JSON, "r") as f:
    meta = json.load(f)

ALPHA = float(meta["alpha"])
T = float(meta["temperature_T"])
V_MIN = float(meta["variance_min"])
V_MAX = float(meta["variance_max"])

qhat_vanilla = float(meta["pixel_level"]["vanilla"]["qhat"])

rules_available = meta["pixel_level"]["adaptive"].keys()
qhats = {r: {} for r in rules_available}
for r in rules_available:
    for lam_str, pack in meta["pixel_level"]["adaptive"][r].items():
        qhats[r][float(lam_str)] = float(pack["qhat"])

print(f"  alpha={ALPHA:.2f}  T={T:.3f}  Vmin={V_MIN:.3e}  Vmax={V_MAX:.3e}")
print("  Loaded qhat for rules:", ", ".join(rules_available))

# ========================= 2022 points: cache =========================

def maybe_run_or_load_points_2022():
    if (not RUN_2022_INFERENCE) and os.path.isfile(CACHE_2022_NPZ):
        print(f"Loading 2022 cache: {CACHE_2022_NPZ}")
        D = np.load(CACHE_2022_NPZ)
        return D["p0"], D["p1"], D["V_center"], D["y"], D["ids"]

    print("Running ensemble on 2022 points to build cache...")
    ensemble = {}
    for key, emd in MODEL_EMDS.items():
        try:
            m, cfg = load_model_and_config(emd, device=DEVICE)
            ensemble[key] = {"model": m, "config": cfg}
            print(f"  Loaded {key}")
        except Exception as e:
            warnings.warn(f"Failed to load {key}: {e}")
    if not ensemble:
        raise RuntimeError("No models for 2022 inference")

    with fiona.open(POINTS_SHP, "r") as src:
        feats = [(shape(f["geometry"]), f["properties"]) for f in src]
    if MAX_POINTS is not None:
        feats = feats[:MAX_POINTS]
    print(f"  Points: {len(feats)}")

    p0_list, p1_list, v_list, y_list, id_list = [], [], [], [], []
    with rasterio.open(RASTER_PATH) as ds:
        for i, (geom, props) in enumerate(tqdm(feats, desc="Points")):
            x, y_ = geom.x, geom.y
            row, col = ds.index(x, y_)
            chip = safe_read_chip(ds, row, col, CHIP_SIZE)
            if chip is None:
                continue

            per_model_logits = []
            per_model_probs_c1 = []
            for key, pack in ensemble.items():
                try:
                    chip_norm = norm_chip(chip, pack["config"])
                    logits = model_logits(pack["model"], chip_norm, device=DEVICE)
                    per_model_logits.append(logits)
                    z = logits - logits.max(axis=0, keepdims=True)
                    e = np.exp(z)
                    probs = e / np.clip(e.sum(axis=0, keepdims=True), 1e-12, None)
                    per_model_probs_c1.append(probs[min(1, probs.shape[0] - 1)])
                except Exception as e:
                    warnings.warn(f"Inference failed for model {key} at point {i}: {e}")

            if len(per_model_logits) == 0:
                continue

            L_stack = np.stack(per_model_logits, axis=0)
            mean_logits = np.mean(L_stack, axis=0)
            c1_var_map = np.var(np.stack(per_model_probs_c1, axis=0), axis=0)

            cy, cx = CHIP_SIZE // 2, CHIP_SIZE // 2
            logits_center = mean_logits[:, cy, cx]
            probs_center = softmax_vec(logits_center / max(T, 1e-8))

            y_true = props.get(GT_FIELD, None)
            if y_true is None:
                continue
            y_true = int(y_true)

            p0_list.append(float(probs_center[0]))
            p1_list.append(float(probs_center[min(1, len(probs_center) - 1)]))
            v_list.append(float(c1_var_map[cy, cx]))
            y_list.append(y_true)
            id_list.append(i)

    p0 = np.array(p0_list, dtype=np.float32)
    p1 = np.array(p1_list, dtype=np.float32)
    Vc = np.array(v_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int16)
    ids = np.array(id_list, dtype=np.int64)

    np.savez_compressed(CACHE_2022_NPZ, p0=p0, p1=p1, V_center=Vc, y=y, ids=ids)
    print(f"Saved 2022 cache: {CACHE_2022_NPZ}  (N={len(ids)})")
    return p0, p1, Vc, y, ids


p0, p1, Vc, y, ids = maybe_run_or_load_points_2022()
N = len(ids)
print(f"N points: {N}")

# ========================= Build linear variance & base scores =========================
Vn = (Vc - V_MIN) / max(V_MAX - V_MIN, 1e-8)
Vn = np.clip(Vn, 0.0, 1.0)

py = np.where(y == 1, p1, p0)
base_scores = 1.0 - py

# ========================= Evaluate vanilla + linear =========================

def evaluate_method(q, scores_method):
    covered = (scores_method <= q)
    cov = float(np.mean(covered)) if len(covered) else float("nan")
    pc = {}
    for c in [0, 1]:
        idx = (y == c)
        pc[c] = float(np.mean(scores_method[idx] <= q)) if np.any(idx) else np.nan
    return cov, pc


def summarize_setcomp(q, rule=None, lam=None):
    if lam is None or rule is None:
        return set_composition_binary(p0, p1, q)
    transform = lambda s: s / (1.0 + lam * Vn)
    return set_composition_binary(p0, p1, q, transform=transform)


summary_rows = []
methods_summary = {}

# Vanilla
qv = qhat_vanilla
cov_v, pc_v = evaluate_method(qv, base_scores)
sc_v = summarize_setcomp(qv)
methods_summary["vanilla"] = {
    "qhat": qv,
    "coverage": cov_v,
    "per_class_coverage": pc_v,
    "set_composition": sc_v,
}
summary_rows.append(["vanilla", "-", qv, cov_v, pc_v.get(0, np.nan), pc_v.get(1, np.nan),
                     sc_v["pct_single"], sc_v["pct_empty"], sc_v["pct_two"]])

# Linear rule only
RULE = "linear"
if RULE not in qhats:
    raise RuntimeError("Linear rule not found in META_JSON adaptive section.")
methods_summary[RULE] = {}
for lam in COMPARE_LAMBDAS:
    if lam not in qhats[RULE]:
        warnings.warn(f"No qhat for linear lambda={lam} in meta; skipping.")
        continue
    q = qhats[RULE][lam]
    scores_m = base_scores / (1.0 + lam * Vn)
    cov, pc = evaluate_method(q, scores_m)
    sc = summarize_setcomp(q, rule=RULE, lam=lam)
    methods_summary[RULE][str(lam)] = {
        "qhat": q,
        "coverage": cov,
        "per_class_coverage": pc,
        "set_composition": sc,
    }
    summary_rows.append([RULE, lam, q, cov, pc.get(0, np.nan), pc.get(1, np.nan),
                         sc["pct_single"], sc["pct_empty"], sc["pct_two"]])

# Save per-point CSV
per_point = pd.DataFrame({
    "id": ids,
    "gt": y,
    "p0": p0,
    "p1": p1,
    "V_center": Vc,
    "Vn": Vn,
    "base_score": base_scores,
    "covered_vanilla": (base_scores <= qhat_vanilla).astype(np.int32),
})

for lam in COMPARE_LAMBDAS:
    if lam not in qhats[RULE]:
        continue
    q = qhats[RULE][lam]
    s = base_scores / (1.0 + lam * Vn)
    per_point[f"{RULE}_score_lam{lam}"] = s
    per_point[f"{RULE}_covered_lam{lam}"] = (s <= q).astype(np.int32)

csv_points = os.path.join(OUT_DIR, "points_with_all_methods.csv")
per_point.to_csv(csv_points, index=False)

comparison = {
    "alpha": ALPHA,
    "temperature_T": T,
    "variance_norm": {"min": V_MIN, "max": V_MAX},
    "counts": {"N_points": int(N), "n_class0": int(np.sum(y == 0)), "n_class1": int(np.sum(y == 1))},
    "methods": methods_summary,
    "files": {"per_point_csv": Path(csv_points).name},
}
summary_json = os.path.join(OUT_DIR, "cp_eval_comparison_2022.json")
with open(summary_json, "w") as f:
    json.dump(comparison, f, indent=2)

print("\nComparison on 2022 points")

def as_pct(x):
    return f"{x*100:.1f}%" if np.isfinite(x) else "nan"

print(f"Vanilla: cov={comparison['methods']['vanilla']['coverage']:.3f} | "
      f"set: single={as_pct(comparison['methods']['vanilla']['set_composition']['pct_single'])} "
      f"empty={as_pct(comparison['methods']['vanilla']['set_composition']['pct_empty'])} "
      f"two={as_pct(comparison['methods']['vanilla']['set_composition']['pct_two'])}")

if RULE in comparison["methods"]:
    for lam_key in sorted(comparison["methods"][RULE].keys(), key=float):
        m = comparison["methods"][RULE][lam_key]
        sc = m["set_composition"]
        print(f"{RULE:10s} lambda={float(lam_key):.1f} | cov={m['coverage']:.3f} | "
              f"set: single={as_pct(sc['pct_single'])} empty={as_pct(sc['pct_empty'])} two={as_pct(sc['pct_two'])} | "
              f"class0={m['per_class_coverage'].get(0, float('nan')):.3f}  class1={m['per_class_coverage'].get(1, float('nan')):.3f}")

print(f"\nSaved per-point CSV: {csv_points}")
print(f"Saved summary JSON : {summary_json}")

# ========================= Figures =========================
print("\nCreating figures...")

lam_present = sorted([float(k) for k in comparison["methods"].get(RULE, {}).keys()])

if lam_present:
    coverage = [comparison["methods"][RULE][str(l)]["coverage"] for l in lam_present]
    singleton = [comparison["methods"][RULE][str(l)]["set_composition"]["pct_single"] * 100 for l in lam_present]
    van_cov = comparison["methods"]["vanilla"]["coverage"]
    van_single = comparison["methods"]["vanilla"]["set_composition"]["pct_single"] * 100

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax2 = ax1.twinx()

    ax1.plot(lam_present, coverage, "o-", linewidth=2, label="Coverage", color="tab:blue")
    ax2.plot(lam_present, singleton, "s--", linewidth=2, label="Singleton %", color="tab:orange")

    ax1.axhline(van_cov, color="tab:blue", ls=":", lw=1.8, alpha=0.85)
    ax2.axhline(van_single, color="tab:orange", ls=":", lw=1.8, alpha=0.85)

    ax1.set_xlabel("lambda")
    ax1.set_ylabel("Coverage", color="tab:blue")
    ax2.set_ylabel("Singleton percentage (%)", color="tab:orange")
    ax1.set_title("Coverage & Singleton vs lambda\n(Vanilla shown as dotted reference)")
    ax1.grid(alpha=0.35)

    fig.tight_layout()
    fig1_path = os.path.join(OUT_DIR, "fig_linear_coverage_singleton_vs_lambda.png")
    fig.savefig(fig1_path, dpi=200)
    plt.close(fig)

if lam_present:
    class0_cov = [comparison["methods"][RULE][str(l)]["per_class_coverage"][0] for l in lam_present]
    class1_cov = [comparison["methods"][RULE][str(l)]["per_class_coverage"][1] for l in lam_present]
    v0 = comparison["methods"]["vanilla"]["per_class_coverage"][0]
    v1 = comparison["methods"]["vanilla"]["per_class_coverage"][1]

    plt.figure(figsize=(7, 4.5))
    plt.plot(lam_present, class0_cov, "o-", linewidth=2, label="Class 0", color="tab:green")
    plt.plot(lam_present, class1_cov, "s--", linewidth=2, label="Class 1", color="tab:red")
    plt.axhline(v0, color="tab:green", ls=":", lw=1.8, alpha=0.85)
    plt.axhline(v1, color="tab:red", ls=":", lw=1.8, alpha=0.85)
    plt.xlabel("lambda")
    plt.ylabel("Per-class coverage")
    plt.title("Per-class Coverage vs lambda (Linear Rule)\n(Vanilla shown as dotted reference)")
    plt.legend()
    plt.grid(alpha=0.35)
    plt.tight_layout()
    fig2_path = os.path.join(OUT_DIR, "fig_linear_per_class_coverage_vs_lambda.png")
    plt.savefig(fig2_path, dpi=200)
    plt.close()

lam_show = 3.0
if str(lam_show) in comparison["methods"].get(RULE, {}):
    q_van = comparison["methods"]["vanilla"]["qhat"]
    q_linear = comparison["methods"][RULE][str(lam_show)]["qhat"]
    x = per_point["base_score"].values
    y_sc = per_point[f"{RULE}_score_lam{lam_show}"].values

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y_sc, s=10, alpha=0.45)
    plt.axvline(q_van, color="tab:blue", ls="--", lw=1.8, label="Vanilla qhat")
    plt.axhline(q_linear, color="tab:orange", ls="--", lw=1.8, label=f"Linear qhat (lambda={lam_show:g})")
    lim = [min(x.min(), y_sc.min()), max(x.max(), y_sc.max())]
    plt.plot(lim, lim, color="grey", ls=":", lw=1.2, alpha=0.8)
    plt.xlim(lim)
    plt.ylim(lim)
    plt.xlabel("Vanilla base score  (s = 1 - p_y)")
    plt.ylabel(f"Linear score (s' = s / (1 + lambda Var_n), lambda={lam_show:g})")
    plt.title("Per-point Score Transformation: Vanilla vs Linear (lambda=3)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    fig3_path = os.path.join(OUT_DIR, "fig_scatter_vanilla_vs_linear_lambda3.png")
    plt.savefig(fig3_path, dpi=220)
    plt.close()

if lam_present:
    coverage = [comparison["methods"][RULE][str(l)]["coverage"] for l in lam_present]
    two_pct = [comparison["methods"][RULE][str(l)]["set_composition"]["pct_two"] * 100 for l in lam_present]

    plt.figure(figsize=(7, 4.5))
    plt.plot(two_pct, coverage, "o-", linewidth=2, color="tab:purple")
    for i, l in enumerate(lam_present):
        plt.annotate(f"lambda={l:g}", (two_pct[i], coverage[i]), textcoords="offset points", xytext=(5, 5), fontsize=8)
    plt.xlabel("Two-label set percentage (%)")
    plt.ylabel("Coverage")
    plt.title("Coverage vs Two-label Frequency (Linear Rule)")
    plt.grid(alpha=0.35)
    plt.tight_layout()
    fig4_path = os.path.join(OUT_DIR, "fig_linear_coverage_vs_two_label.png")
    plt.savefig(fig4_path, dpi=200)
    plt.close()

print("\nFigures written to:", OUT_DIR)
