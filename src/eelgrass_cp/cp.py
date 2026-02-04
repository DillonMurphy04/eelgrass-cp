from typing import Dict, Optional

import numpy as np


def qhat_conformal(scores_cal: np.ndarray, alpha: float) -> float:
    """Finite-sample conformal quantile: ceil((n+1)*(1-alpha))/n."""
    s = np.sort(np.asarray(scores_cal))
    n = len(s)
    if n == 0:
        return float("nan")
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(max(k, 1), n)
    return float(s[k - 1])


def qhat_split_conformal(scores: np.ndarray, alpha: float, cal_idx: np.ndarray) -> float:
    return qhat_conformal(scores[cal_idx], alpha)


def set_composition_binary(p0: np.ndarray, p1: np.ndarray, q: float,
                           transform: Optional[callable] = None) -> Dict[str, float]:
    s0 = 1 - p0
    s1 = 1 - p1
    if transform is not None:
        s0 = transform(s0)
        s1 = transform(s1)
    k = (s0 <= q).astype(np.int32) + (s1 <= q).astype(np.int32)
    n = len(k)
    e = int(np.sum(k == 0))
    s = int(np.sum(k == 1))
    t = n - e - s
    return {
        "n": n,
        "empty": e,
        "single": s,
        "two": t,
        "pct_empty": e / n if n else np.nan,
        "pct_single": s / n if n else np.nan,
        "pct_two": t / n if n else np.nan,
    }


def per_class_coverage(scores: np.ndarray, y: np.ndarray, q: float) -> Dict[int, float]:
    out = {}
    for c in np.unique(y):
        idx = (y == c)
        out[int(c)] = float(np.mean(scores[idx] <= q)) if np.any(idx) else np.nan
    return out


def binned_coverage(scores: np.ndarray, q: float, vnorm: np.ndarray, nbins: int = 10):
    edges = np.quantile(vnorm, np.linspace(0, 1, nbins + 1))
    edges[0], edges[-1] = 0.0, 1.0
    covs, mids, counts = [], [], []
    for i in range(nbins):
        m = (vnorm >= edges[i]) & ((vnorm < edges[i + 1]) if i < nbins - 1 else (vnorm <= edges[i + 1]))
        if m.sum() == 0:
            covs.append(np.nan)
            mids.append(0.5 * (edges[i] + edges[i + 1]))
            counts.append(0)
        else:
            covs.append(float(np.mean(scores[m] <= q)))
            mids.append(0.5 * (edges[i] + edges[i + 1]))
            counts.append(int(m.sum()))
    return np.array(mids), np.array(covs), np.array(counts)

