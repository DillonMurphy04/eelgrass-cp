"""
Sensitivity analysis with bootstrap CIs for CP methods (vanilla, linear lambda=3, nonparametric).
Inputs: 2021 calibration cache + 2022 points cache (or run inference).
Outputs: summary CSV/JSON, bootstrap CI CSV, and figures in OUT_DIR.
"""

import os, json, warnings
from pathlib import Path

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import rasterio
import fiona
from shapely.geometry import shape
from tqdm import tqdm

from src.eelgrass_cp import (
    load_model_and_config,
    norm_chip,
    model_logits,
    softmax_rows,
    softmax_vec,
    safe_read_chip,
    normalize_var,
    fit_sigma_fn_from_cal,
    set_composition_binary,
    per_class_coverage,
    binned_coverage,
    qhat_conformal,
    linear_score_transform,
    corr,
)

# ------------------------- User Config -------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Roots
CAL_BASE_DIR = r"D:\4 year training data\2021\ensemble_2020_2021_pixel_calib_cache"
CACHE_2021_NPZ = os.path.join(CAL_BASE_DIR, "pixel_calib_cache_2021.npz")
META_JSON = os.path.join(CAL_BASE_DIR, "pixel_calib_meta_2021.json")

RASTER_2022 = r"D:\2022 Data\Morro Bay Eelgrass AI - Rasterized Imagery\2022_raster_final.tif"
POINTS_SHP = r"D:\2022 Data\2022points\2022points.shp"

OUT_DIR = r"D:\2022 Data\cp_sensitivity_main"
os.makedirs(OUT_DIR, exist_ok=True)
CACHE_2022_NPZ = os.path.join(OUT_DIR, "points_2022_cache_for_sensitivity.npz")
RUN_2022_INFERENCE = False  # set True to rebuild 2022 cache

# Ensemble EMDs (only needed if RUN_2022_INFERENCE=True)
MODEL_EMDS = {
    "unet_4yr": r"D:\4yr_Unet_model\models\checkpoint_2025-09-19_22-47-55_epoch_11\checkpoint_2025-09-19_22-47-55_epoch_11.emd",
    "samlora_4yr": r"D:\sam_lora_model\samlora_4_year\models\checkpoint_2025-06-11_18-24-03_epoch_18\checkpoint_2025-06-11_18-24-03_epoch_18.emd",
    "samlora_2021": r"D:\sam_lora_model_2021\models\checkpoint_2025-07-30_18-52-20_epoch_18\checkpoint_2025-07-30_18-52-20_epoch_18.emd",
    "deeplab_2021": r"D:\DeepLab_2021\DeepLab_2021.emd",
    "unet_2021": r"D:\2021 Unet\checkpoint_2025-06-25_12-39-26_epoch_11.emd",
    "deeplab_4yr": r"D:\deeplab_4_year\deeplab_4_year.emd",
}

# Misc
ALPHAS = [0.10, 0.05, 0.025]
BINS_SIGMA = 20  # quantile bins for sigma_hat(V)
BINS_COND = 10   # variance deciles for conditional coverage
CHIP_SIZE = 448
GT_FIELD = "GROUND_TRU"
FIG_DPI = 220
LAMBDA_LINEAR = 3.0

# Bootstrap
BOOTSTRAP_B = 1000
BOOTSTRAP_SEED = 123
CI_ALPHA = 0.95  # 95% CI

# Human-readable method labels for outputs/figures
METHOD_LABELS = {
    "vanilla": "Vanilla",
    "linear": "Linear (lambda=3)",
    "nonparametric": "Nonparametric",
}

# ------------------------- Data Loading -------------------------

def load_2021_pool_and_meta():
    if not os.path.isfile(CACHE_2021_NPZ):
        raise FileNotFoundError(f"Missing 2021 cache: {CACHE_2021_NPZ}")
    if not os.path.isfile(META_JSON):
        raise FileNotFoundError(f"Missing 2021 meta: {META_JSON}")

    D2021 = np.load(CACHE_2021_NPZ, allow_pickle=True)
    with open(META_JSON, "r") as f:
        meta = json.load(f)

    T = float(meta["temperature_T"])  # use recorded temperature
    V_MIN = float(meta["variance_min"])
    V_MAX = float(meta["variance_max"])
    SEED = int(meta.get("seed", 42))
    CAL_TEST_SPLIT = float(meta["splits"]["CAL_TEST_SPLIT"])
    CAL_TUNE_FRAC = float(meta["splits"]["CAL_TUNE_FRAC"])  # fraction of cal used to *fit qhat*

    PLOG_all = D2021["PLOG"]  # [N,C]
    Y_all = D2021["Y"]
    V_all = D2021["V"]
    idx_cp = D2021.get("idx_cp", None)
    pool = np.arange(len(Y_all)) if idx_cp is None else idx_cp

    P_cp = softmax_rows(PLOG_all[pool], T=T)
    Y_cp = Y_all[pool]
    V_cp = V_all[pool]

    # Recreate cal/test and cal_sub
    idx_all = np.arange(len(P_cp))
    cal_idx, test_idx = train_test_split(idx_all, test_size=CAL_TEST_SPLIT, random_state=SEED, shuffle=True)
    cal_sub_idx, _ = train_test_split(cal_idx, test_size=CAL_TUNE_FRAC, random_state=SEED, shuffle=True)

    Vn_cp = normalize_var(V_cp, V_MIN, V_MAX)

    return {
        "meta": meta,
        "T": T,
        "V_MIN": V_MIN,
        "V_MAX": V_MAX,
        "P_cp": P_cp,
        "Y_cp": Y_cp,
        "V_cp": V_cp,
        "Vn_cp": Vn_cp,
        "cal_idx": cal_idx,
        "cal_sub_idx": cal_sub_idx,
        "test_idx": test_idx,
    }

def maybe_run_or_load_points_2022(T):
    if (not RUN_2022_INFERENCE) and os.path.isfile(CACHE_2022_NPZ):
        D = np.load(CACHE_2022_NPZ)
        return D["p0"], D["p1"], D["V_center"], D["y"], D["ids"]

    # Build cache via ensemble (center-pixel probs + ensemble variance of class-1 prob)
    ensemble = {}
    for key, emd in MODEL_EMDS.items():
        try:
            m, cfg = load_model_and_config(emd, device=DEVICE, chip_size=CHIP_SIZE)
            ensemble[key] = {"model": m, "config": cfg}
            print(f"Loaded {key}")
        except Exception as e:
            warnings.warn(f"Failed to load {key}: {e}")

    if not ensemble:
        raise RuntimeError("No models available to run 2022 inference")

    with fiona.open(POINTS_SHP, "r") as src:
        feats = [(shape(f["geometry"]), f["properties"]) for f in src]

    p0_list, p1_list, v_list, y_list, id_list = [], [], [], [], []

    with rasterio.open(RASTER_2022) as ds:
        for i, (geom, props) in enumerate(tqdm(feats, desc="Build2022Cache")):
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
                    logits = model_logits(pack["model"], chip_norm, device=DEVICE)
                    per_model_logits.append(logits)

                    z = logits - logits.max(axis=0, keepdims=True)
                    e = np.exp(z); probs = e / np.clip(e.sum(axis=0, keepdims=True), 1e-12, None)
                    per_model_probs_c1.append(probs[min(1, probs.shape[0]-1)])
                except Exception as e:
                    warnings.warn(f"Inference failed for {key} at {i}: {e}")

            if len(per_model_logits) == 0:
                continue

            L_stack = np.stack(per_model_logits, axis=0)
            mean_logits = np.mean(L_stack, axis=0)
            var_map = np.var(np.stack(per_model_probs_c1, axis=0), axis=0)

            cy, cx = CHIP_SIZE//2, CHIP_SIZE//2
            logits_center = mean_logits[:, cy, cx]
            probs_center = softmax_vec(logits_center / max(T, 1e-8))

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
    Vc = np.array(v_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int16)
    ids= np.array(id_list, dtype=np.int64)

    np.savez_compressed(CACHE_2022_NPZ, p0=p0, p1=p1, V_center=Vc, y=y, ids=ids)
    print(f"Saved 2022 cache: {CACHE_2022_NPZ} (N={len(ids)})")
    return p0, p1, Vc, y, ids

# ------------------------- sigma_hat(V) + score transforms -------------------------

# ------------------------- Evaluation helpers -------------------------
def eval_one_method(scores, q, y, p0, p1, transform_for_sets=None, vnorm=None):
    covered = (scores <= q).astype(np.int8)
    cov_overall = float(np.mean(covered)) if len(covered) else float("nan")

    per_class = per_class_coverage(scores, y, q)

    sc = set_composition_binary(p0, p1, q, transform=transform_for_sets, return_k=True)

    cond_stats = None
    if vnorm is not None:
        mids, covs, cnts = binned_coverage(scores, q, vnorm, nbins=BINS_COND)
        r = corr(mids[np.isfinite(covs)], covs[np.isfinite(covs)])
        cond_stats = {"mids": mids.tolist(), "covs": covs.tolist(), "counts": cnts.tolist(), "corr_cov_vs_var": r}

    # also return raw components for bootstrap
    out_raw = {
        "covered": covered,
        "y": y,
        "k_sets": sc["k"],  # 0/1/2
        "vnorm": vnorm if vnorm is not None else None,
        "scores": scores
    }
    return cov_overall, per_class, sc, cond_stats, out_raw

# ------------------------- Bootstrap helpers -------------------------

def _quantile_ci(samples, level=CI_ALPHA):
    lo = (1.0 - level)/2.0
    hi = 1.0 - lo
    return (np.nanquantile(samples, lo), np.nanquantile(samples, hi))

def bootstrap_metrics(raw, target_cov, nbins=BINS_COND, B=BOOTSTRAP_B, seed=BOOTSTRAP_SEED):
    """
    raw: dict from eval_one_method (covered, y, k_sets, vnorm, scores)
    Returns dict of bootstrap mean & CIs for:
      - coverage_overall
      - coverage_class0, coverage_class1
      - pct_empty, pct_single, pct_two
      - cond_corr_cov_vs_var
      - cond_mean_abs_dev_from_target
    """
    rng = np.random.default_rng(seed)
    n = len(raw["covered"])
    if n == 0:
        return {}

    covered = raw["covered"]
    y = raw["y"]
    k = raw["k_sets"]
    vnorm = raw["vnorm"]
    scores = raw["scores"]

    # Precompute bin edges for conditional coverage so each bootstrap uses same bins
    if vnorm is not None:
        edges = np.quantile(vnorm, np.linspace(0,1,nbins+1))
        edges[0], edges[-1] = 0.0, 1.0

    cov_samples = []
    cov0_samples, cov1_samples = [], []
    e_samples, s_samples, t_samples = [], [], []
    corr_samples, mad_samples = [], []

    for _ in range(B):
        idx = rng.integers(0, n, n)  # resample rows with replacement
        c = covered[idx]
        yy = y[idx]
        kk = k[idx]

        # overall coverage
        cov_samples.append(np.mean(c))

        # per-class (guard against empty)
        m0 = (yy == 0); m1 = (yy == 1)
        cov0_samples.append(np.mean(c[m0]) if np.any(m0) else np.nan)
        cov1_samples.append(np.mean(c[m1]) if np.any(m1) else np.nan)

        # set composition
        e_samples.append(np.mean(kk == 0))
        s_samples.append(np.mean(kk == 1))
        t_samples.append(np.mean(kk == 2))

        # conditional pieces
        if vnorm is not None:
            vn = vnorm[idx]
            # reuse 'c' (covered) inside bins with fixed edges
            covs = []
            mids = []
            for i in range(nbins):
                m = (vn >= edges[i]) & ((vn < edges[i+1]) if i < nbins-1 else (vn <= edges[i+1]))
                mids.append(0.5*(edges[i]+edges[i+1]))
                covs.append(np.mean(c[m]) if np.any(m) else np.nan)
            covs = np.array(covs, float)
            mids = np.array(mids, float)
            r = corr(mids[np.isfinite(covs)], covs[np.isfinite(covs)])
            corr_samples.append(r)
            with np.errstate(invalid='ignore'):
                mad_samples.append(np.nanmean(np.abs(covs - target_cov)))
        else:
            corr_samples.append(np.nan); mad_samples.append(np.nan)

    def pack(mean_val, arr):
        lo, hi = _quantile_ci(np.array(arr, float), level=CI_ALPHA)
        return {"mean": float(np.nanmean(arr)), "ci_lo": float(lo), "ci_hi": float(hi)}

    return {
        "coverage_overall": pack(np.mean(cov_samples), cov_samples),
        "coverage_class0": pack(np.nanmean(cov0_samples), cov0_samples),
        "coverage_class1": pack(np.nanmean(cov1_samples), cov1_samples),
        "pct_empty": pack(np.mean(e_samples), e_samples),
        "pct_single": pack(np.mean(s_samples), s_samples),
        "pct_two": pack(np.mean(t_samples), t_samples),
        "cond_corr_cov_vs_var": pack(np.nanmean(corr_samples), corr_samples),
        "cond_mean_abs_dev_from_target": pack(np.nanmean(mad_samples), mad_samples),
        "B": B,
        "ci_level": CI_ALPHA
    }

# ------------------------- Main -------------------------

if __name__ == "__main__":
    # Load 2021 CP pool + rebuild splits
    Z = load_2021_pool_and_meta()
    T, V_MIN, V_MAX = Z["T"], Z["V_MIN"], Z["V_MAX"]
    P_cp, Y_cp, V_cp, Vn_cp = Z["P_cp"], Z["Y_cp"], Z["V_cp"], Z["Vn_cp"]
    cal_idx, cal_sub_idx, test_idx = Z["cal_idx"], Z["cal_sub_idx"], Z["test_idx"]

    # Base scores on CP pool
    py_cp = P_cp[np.arange(len(P_cp)), Y_cp]
    s_base_cp = 1.0 - py_cp

    # 2021 TEST arrays for evaluation
    P_test = P_cp[test_idx]
    Y_test = Y_cp[test_idx]
    Vn_test = Vn_cp[test_idx]
    s_test = s_base_cp[test_idx]
    p0_test = P_test[:,0]
    p1_test = P_test[:,min(1, P_test.shape[1]-1)]

    # sigma_hat(V) from CAL (full cal_idx)
    sigma_fn, (sigma_x, sigma_y) = fit_sigma_fn_from_cal(Vn_cp, s_base_cp, cal_idx, bins=BINS_SIGMA)

    # 2022 cache
    p0_22, p1_22, Vc_22, y_22, ids_22 = maybe_run_or_load_points_2022(T)
    Vn_22 = normalize_var(Vc_22, V_MIN, V_MAX)
    py_22 = np.where(y_22 == 1, p1_22, p0_22)
    s_22 = 1.0 - py_22

    # Figures dir
    figs_dir = os.path.join(OUT_DIR, "figs"); os.makedirs(figs_dir, exist_ok=True)

    # Save sigma_hat(V) figure (once)
    plt.figure(figsize=(6.8,4.2))
    plt.plot(sigma_x, sigma_y, "o-", lw=2)
    plt.xlabel("Normalized ensemble variance Vn")
    plt.ylabel("sigma_hat(V) = E[1 - p_y | Vn]")
    plt.title("Nonparametric scale from 2021 calibration")
    plt.grid(alpha=0.35); plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "sigma_curve.png"), dpi=FIG_DPI); plt.close()

    # Prepare results collectors
    rows_summary = []   # flat metrics table
    rows_boot = []      # bootstrap CI table (long-form)

    # Helper to map method -> transform and score arrays per dataset
    def get_method_defs(vnorm_test, vnorm_22):
        return {
            "vanilla": {
                "score_test": s_test,
                "score_22": s_22,
                "set_transform_test": None,
                "set_transform_22": None,
            },
            "linear": {
                "score_test": linear_score_transform(s_test, vnorm_test, LAMBDA_LINEAR),
                "score_22": linear_score_transform(s_22, vnorm_22, LAMBDA_LINEAR),
                "set_transform_test": (lambda s: linear_score_transform(s, vnorm_test, LAMBDA_LINEAR)),
                "set_transform_22": (lambda s: linear_score_transform(s, vnorm_22, LAMBDA_LINEAR)),
            },
            "nonparametric": {
                "score_test": s_test / sigma_fn(vnorm_test),
                "score_22": s_22 / sigma_fn(vnorm_22),
                "set_transform_test": (lambda s: s / sigma_fn(vnorm_test)),
                "set_transform_22": (lambda s: s / sigma_fn(vnorm_22)),
            },
        }

    method_defs = get_method_defs(Vn_test, Vn_22)

    def qhat_for_method(method_key, alpha):
        if method_key == "vanilla":
            s_cal_sub = s_base_cp[cal_sub_idx]
        elif method_key == "linear":
            s_cal_sub = linear_score_transform(s_base_cp[cal_sub_idx], Vn_cp[cal_sub_idx], LAMBDA_LINEAR)
        elif method_key == "nonparametric":
            s_cal_sub = s_base_cp[cal_sub_idx] / sigma_fn(Vn_cp[cal_sub_idx])
        else:
            raise ValueError(method_key)
        return qhat_conformal(s_cal_sub, alpha)

    # Loop over alphas and evaluate on both datasets
    for alpha in ALPHAS:
        target = 1 - alpha
        for method_key in ["vanilla", "linear", "nonparametric"]:
            label = METHOD_LABELS[method_key]

            # qhat
            qhat = qhat_for_method(method_key, alpha)

            # 2021 TEST
            sT = method_defs[method_key]["score_test"]
            covT, pcT, scT, condT, rawT = eval_one_method(
                sT, qhat, Y_test, p0_test, p1_test,
                transform_for_sets=method_defs[method_key]["set_transform_test"],
                vnorm=Vn_test,
            )
            if condT is not None:
                condT["target"] = target
                covs = np.array(condT["covs"], float)
                with np.errstate(invalid='ignore'):
                    condT["mean_abs_dev_from_target"] = float(np.nanmean(np.abs(covs - target)))

            # 2022 OOD
            sO = method_defs[method_key]["score_22"]
            covO, pcO, scO, condO, rawO = eval_one_method(
                sO, qhat, y_22, p0_22, p1_22,
                transform_for_sets=method_defs[method_key]["set_transform_22"],
                vnorm=Vn_22,
            )
            if condO is not None:
                condO["target"] = target
                covs = np.array(condO["covs"], float)
                with np.errstate(invalid='ignore'):
                    condO["mean_abs_dev_from_target"] = float(np.nanmean(np.abs(covs - target)))

            # Append rows for a flat CSV table
            def pack(domain, cov, pc, sc, cond):
                return {
                    "method_key": method_key,
                    "method": label,
                    "alpha": alpha,
                    "target_cov": target,
                    "domain": domain,  # "2021_test" or "2022_OOD"
                    "coverage_overall": cov,
                    "coverage_class0": pc.get(0, np.nan),
                    "coverage_class1": pc.get(1, np.nan),
                    "pct_empty": sc["pct_empty"],
                    "pct_single": sc["pct_single"],
                    "pct_two": sc["pct_two"],
                    "cond_corr_cov_vs_var": (cond or {}).get("corr_cov_vs_var", np.nan),
                    "cond_mean_abs_dev_from_target": (cond or {}).get("mean_abs_dev_from_target", np.nan),
                    "qhat": qhat
                }

            rows_summary.append(pack("2021_test", covT, pcT, scT, condT))
            rows_summary.append(pack("2022_OOD",  covO, pcO, scO, condO))

            # ---------- Bootstrap CIs ----------
            bootT = bootstrap_metrics(rawT, target_cov=target, nbins=BINS_COND, B=BOOTSTRAP_B, seed=BOOTSTRAP_SEED)
            bootO = bootstrap_metrics(rawO, target_cov=target, nbins=BINS_COND, B=BOOTSTRAP_B, seed=BOOTSTRAP_SEED+1)

            def rows_from_boot(boot, domain):
                if not boot:
                    return []
                out = []
                for metric, blob in boot.items():
                    if metric in ("B", "ci_level"):
                        continue
                    out.append({
                        "method_key": method_key,
                        "method": label,
                        "alpha": alpha,
                        "domain": domain,
                        "metric": metric,
                        "mean": blob["mean"],
                        "ci_lo": blob["ci_lo"],
                        "ci_hi": blob["ci_hi"],
                        "B": boot.get("B", BOOTSTRAP_B),
                        "ci_level": boot.get("ci_level", CI_ALPHA),
                        "qhat": qhat
                    })
                return out

            rows_boot.extend(rows_from_boot(bootT, "2021_test"))
            rows_boot.extend(rows_from_boot(bootO, "2022_OOD"))

            # --- Figures per alpha/method ---
            # Coverage vs variance (for both domains) in one panel
            def plot_cov_vs_var(mids, covs, label_txt):
                plt.plot(mids, covs, marker="o", lw=2, label=label_txt)

            plt.figure(figsize=(7.8,4.6))
            if condT is not None:
                mT, cT = np.array(condT["mids"]), np.array(condT["covs"])
                plot_cov_vs_var(mT, cT, "2021 test")
            if condO is not None:
                mO, cO = np.array(condO["mids"]), np.array(condO["covs"])
                plot_cov_vs_var(mO, cO, "2022 OOD")
            plt.axhline(target, color="k", ls=":", lw=1.1)
            plt.ylim(0,1)
            plt.xlabel("Normalized variance (bin mid)")
            plt.ylabel("Coverage per bin")
            plt.title(f"Coverage vs Variance -- {label} -- alpha={alpha}")
            plt.grid(alpha=0.35); plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(figs_dir, f"cov_vs_var__{method_key}__alpha{alpha}.png"), dpi=FIG_DPI)
            plt.close()

            # Set composition bars (both domains)
            def bar(ax, sc, title_txt):
                ax.bar(["empty","single","two"], [sc["pct_empty"], sc["pct_single"], sc["pct_two"]])
                ax.set_ylim(0,1); ax.set_title(title_txt)

            fig, ax = plt.subplots(1,2, figsize=(8.8,3.8), sharey=True)
            bar(ax[0], scT, "2021 test")
            bar(ax[1], scO, "2022 OOD")
            fig.supylabel("fraction of test points")
            fig.suptitle(f"Set composition -- {label} -- alpha={alpha}")
            plt.tight_layout()
            plt.savefig(os.path.join(figs_dir, f"set_comp__{method_key}__alpha{alpha}.png"), dpi=FIG_DPI)
            plt.close()

    # ---------- Save tables ----------
    df_summary = pd.DataFrame(rows_summary)
    csv_path = os.path.join(OUT_DIR, "sensitivity_summary.csv")
    df_summary.to_csv(csv_path, index=False)

    # Bootstrap table
    df_boot = pd.DataFrame(rows_boot)
    boot_path = os.path.join(OUT_DIR, "sensitivity_bootstrap_ci.csv")
    df_boot.to_csv(boot_path, index=False)

    # Also pivot for a pretty appendix table (one per domain)
    for domain in ["2021_test", "2022_OOD"]:
        df_dom = df_summary[df_summary["domain"]==domain]
        piv = df_dom.pivot_table(
            index=["method","alpha"],
            values=[
                "coverage_overall","coverage_class0","coverage_class1",
                "pct_empty","pct_single","pct_two",
                "cond_corr_cov_vs_var","cond_mean_abs_dev_from_target","qhat"
            ],
        )
        piv.sort_index().to_csv(os.path.join(OUT_DIR, f"sensitivity_pivot_{domain}.csv"))

    # JSON summary with metadata
    with open(META_JSON, "r") as f:
        meta_json = json.load(f)

    summary = {
        "alphas": ALPHAS,
        "bins": {"sigma": BINS_SIGMA, "conditional": BINS_COND},
        "variance_norm_minmax": {"min": float(V_MIN), "max": float(V_MAX)},
        "n_2021_test": int(len(test_idx)),
        "n_2022": int(len(y_22)),
        "bootstrap": {
            "B": BOOTSTRAP_B,
            "ci_level": CI_ALPHA,
            "file": Path(boot_path).name
        },
        "files": {
            "csv_flat": Path(csv_path).name,
            "csv_pivot_2021_test": f"sensitivity_pivot_2021_test.csv",
            "csv_pivot_2022_OOD": f"sensitivity_pivot_2022_OOD.csv",
            "fig_dir": Path(figs_dir).name,
        },
        "notes": {
            "qhat_rule": "ceil((n+1)*(1-alpha))/n finite-sample split conformal quantile on calibration subset",
            "coverage_target": {str(a): 1.0 - a for a in ALPHAS}
        }
    }
    with open(os.path.join(OUT_DIR, "sensitivity_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # -------- Thesis-friendly summary (2022 OOD, alpha = first in ALPHAS) --------
    alpha_main = ALPHAS[0]
    mask_main = (df_summary["domain"] == "2022_OOD") & np.isclose(df_summary["alpha"], alpha_main)
    df_main = df_summary.loc[mask_main, [
        "method",
        "coverage_overall",
        "coverage_class0",
        "coverage_class1",
        "pct_empty",
        "pct_single",
        "pct_two",
        "cond_mean_abs_dev_from_target",
        "cond_corr_cov_vs_var",
    ]].copy()

    print(f"\n=== Summary table (2022 OOD, alpha={alpha_main}) ===")
    if not df_main.empty:
        print(df_main.to_string(index=False, float_format=lambda x: f"{x:0.3f}"))
    else:
        print("No rows found for this alpha/domain (check ALPHAS list).")

    # Also export a compact summary set-composition figure across methods
    if not df_main.empty:
        methods = df_main["method"].tolist()
        empty_vals = df_main["pct_empty"].values
        single_vals = df_main["pct_single"].values
        two_vals = df_main["pct_two"].values

        x = np.arange(len(methods))
        w = 0.25

        plt.figure(figsize=(7.5,4.2))
        plt.bar(x - w, empty_vals, width=w, label="Empty")
        plt.bar(x,       single_vals, width=w, label="Single")
        plt.bar(x + w, two_vals,   width=w, label="Two-label")

        plt.xticks(x, methods, rotation=0)
        plt.ylim(0, 1.0)
        plt.ylabel("Fraction of test points")
        plt.title(f"Set composition by method -- 2022 OOD, alpha={alpha_main}")
        plt.legend()
        plt.tight_layout()
        summary_fig = os.path.join(figs_dir, f"summary_set_comp_2022_alpha{alpha_main}.png")
        plt.savefig(summary_fig, dpi=FIG_DPI)
        plt.close()
        print("Summary set-composition figure:", summary_fig)

    print("\nSensitivity analysis complete.")
    print("->", csv_path)
    print("->", boot_path)
    print("->", os.path.join(OUT_DIR, "sensitivity_pivot_2021_test.csv"))
    print("->", os.path.join(OUT_DIR, "sensitivity_pivot_2022_OOD.csv"))
    print("-> figs in:", figs_dir)
